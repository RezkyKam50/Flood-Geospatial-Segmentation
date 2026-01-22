import argparse
import logging
import sys
import torch
from torch import nn
from models.u_net import UNet
from models.prithvi_segmenter import PritviSegmenter
from models.prithvi_unet import PrithviUNet
import os
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from enum import Enum
from utils.testing import computeIOU, computeAccuracy, computeMetrics
from utils.dice import DiceLoss
from data_loading.sen1floods11 import get_loader
import json

torch.manual_seed(124)

class DatasetType(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    BOLIVIA = 'bolivia'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Compare Prithvi, UNet, and Prithvi-UNet models')
    parser.add_argument('--data_path', type=str, default='./datasets/sen1floods11_v1.1', help='Path to the data directory.')
    parser.add_argument('--version', type=str, default='comparison', help='Experiment version')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for the optimizer.')
    
    # Model-specific parameters
    parser.add_argument('--prithvi_out_channels', type=int, default=768, help='Number of output channels from the Prithvi encoders')
    parser.add_argument('--unet_out_channels', type=int, default=768, help='Number of output channels from the UNet encoders')
    parser.add_argument('--prithvi_finetune_ratio', type=float, default=1, help='Fine-tune ratio for Prithvi models')
    
    # Training parameters
    parser.add_argument('--save_model_interval', type=int, default=5, help='Save the model every n epochs')
    parser.add_argument('--test_interval', type=int, default=1, help='Test the model every n epochs')
    
    # U-Prithvi specific
    parser.add_argument('--combine_func', type=str, default='concat', choices=['concat', 'mul', 'add'], help='Combination function for U-Prithvi')
    parser.add_argument('--random_dropout_prob', type=float, default=2/3, help='Dropout probability for U-Prithvi')
    
    return parser.parse_args()

def get_number_of_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, loader, optimizer, criterion, epoch, device):
    model.train()
    running_loss = 0.0
    running_samples = 0
    running_accuracies = 0
    running_iou = 0
    
    for batch_idx, (imgs, masks) in enumerate(tqdm(loader, desc=f"Training Epoch {epoch+1}"), 0):
        optimizer.zero_grad()
        imgs = imgs.to(device)
        masks = masks.to(device)
        outputs = model(imgs)
        targets = masks.squeeze(1)

        loss = criterion(outputs, targets.long())
        loss.backward()
        
        iou = computeIOU(outputs, targets, device)
        accuracy = computeAccuracy(outputs, targets, device)
        
        optimizer.step()
    
        running_samples += targets.size(0)
        running_loss += loss.item()
        running_accuracies += accuracy
        running_iou += iou
    
    avg_loss = running_loss / (batch_idx + 1)
    avg_acc = running_accuracies / (batch_idx + 1)
    avg_iou = running_iou / (batch_idx + 1)
    
    return avg_loss, avg_acc, avg_iou

def test(model, loader, criterion, device):
    model.eval()
    metricss = {}
    index = 0
    
    with torch.no_grad():
        for (imgs, masks) in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            predictions = model(imgs)
            
            metrics = computeMetrics(predictions, masks, device, criterion)
            metricss = {k: metricss.get(k, 0) + v for k, v in metrics.items()}
            index += 1
    
    TP, FP, TN, FN, loss = metricss['TP'].item(), metricss['FP'].item(), metricss['TN'].item(), metricss['FN'].item(), metricss['loss']
    
    IOU_floods = TP / (TP + FN + FP) if (TP + FN + FP) > 0 else 0
    IOU_non_floods = TN / (TN + FP + FN) if (TN + FP + FN) > 0 else 0
    Avg_IOU = (IOU_floods + IOU_non_floods) / 2

    ACC_floods = TP / (TP + FN) if (TP + FN) > 0 else 0
    ACC_non_floods = TN / (TN + FP) if (TN + FP) > 0 else 0
    Avg_ACC = (ACC_floods + ACC_non_floods) / 2
    
    return {
        'IOU_floods': IOU_floods,
        'IOU_non_floods': IOU_non_floods,
        'Avg_IOU': Avg_IOU,
        'ACC_floods': ACC_floods,
        'ACC_non_floods': ACC_non_floods,
        'Avg_ACC': Avg_ACC,
        'Loss': loss / index
    }

def train_single_model(model_name, model, train_loader, valid_loader, test_loader, bolivia_loader, args, device, base_log_dir):
    """Train a single model and return results"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*80}")
    
    # Setup logging for this model
    model_log_dir = os.path.join(base_log_dir, model_name)
    os.makedirs(model_log_dir, exist_ok=True)
    model_dir = os.path.join(model_log_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    writer = SummaryWriter(model_log_dir)
    
    # Model info
    num_params = get_number_of_trainable_parameters(model)
    logger.info(f"Number of trainable parameters: {num_params:,}")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = DiceLoss(device=device)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, args.epochs)
    
    # Freeze Prithvi if applicable
    if model_name in ['prithvi', 'prithvi_unet']:
        model.change_prithvi_trainability(True)
        logger.info(f"Prithvi weights frozen. Trainable parameters: {get_number_of_trainable_parameters(model):,}")
    
    # Training loop
    for epoch in range(args.epochs):
        logger.info(f"\n{model_name} - Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc, train_iou = train_model(model, train_loader, optimizer, criterion, epoch, device)
        logger.info(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, IoU: {train_iou:.4f}")
        
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("IoU/train", train_iou, epoch)
        
        scheduler.step()
        
        # Validation
        if (epoch + 1) % args.test_interval == 0:
            val_metrics = test(model, valid_loader, criterion, device)
            logger.info(f"Valid - Avg IOU: {val_metrics['Avg_IOU']:.4f}, Avg ACC: {val_metrics['Avg_ACC']:.4f}, Loss: {val_metrics['Loss']:.4f}")
            
            for metric_name, metric_value in val_metrics.items():
                writer.add_scalar(f"{metric_name}/valid", metric_value, epoch)
        
        # Save model
        if (epoch + 1) % args.save_model_interval == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f"model_epoch_{epoch+1}.pt"))
    
    # Fine-tuning for Prithvi models
    if model_name in ['prithvi', 'prithvi_unet'] and args.prithvi_finetune_ratio is not None:
        logger.info(f"\nFine-tuning {model_name}")
        
        finetune_epochs = int(args.epochs * args.prithvi_finetune_ratio)
        model.change_prithvi_trainability(True)
        logger.info(f"Prithvi weights unfrozen. Trainable parameters: {get_number_of_trainable_parameters(model):,}")
        
        # New optimizer with lower learning rate
        finetune_lr = args.learning_rate * 0.1
        optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, finetune_epochs)
        
        for epoch in range(args.epochs, args.epochs + finetune_epochs):
            logger.info(f"\n{model_name} - Fine-tune Epoch {epoch+1}/{args.epochs + finetune_epochs}")
            
            train_loss, train_acc, train_iou = train_model(model, train_loader, optimizer, criterion, epoch, device)
            logger.info(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, IoU: {train_iou:.4f}")
            
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("IoU/train", train_iou, epoch)
            
            scheduler.step()
            
            if (epoch + 1) % args.test_interval == 0:
                val_metrics = test(model, valid_loader, criterion, device)
                logger.info(f"Valid - Avg IOU: {val_metrics['Avg_IOU']:.4f}, Avg ACC: {val_metrics['Avg_ACC']:.4f}")
                
                for metric_name, metric_value in val_metrics.items():
                    writer.add_scalar(f"{metric_name}/valid", metric_value, epoch)
    
    # Final evaluation
    logger.info(f"\n{model_name} - Final Evaluation")
    test_metrics = test(model, test_loader, criterion, device)
    bolivia_metrics = test(model, bolivia_loader, criterion, device)
    
    logger.info(f"Test Set - Avg IOU: {test_metrics['Avg_IOU']:.4f}, Avg ACC: {test_metrics['Avg_ACC']:.4f}, Loss: {test_metrics['Loss']:.4f}")
    logger.info(f"Bolivia Set - Avg IOU: {bolivia_metrics['Avg_IOU']:.4f}, Avg ACC: {bolivia_metrics['Avg_ACC']:.4f}, Loss: {bolivia_metrics['Loss']:.4f}")
    
    writer.close()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(model_dir, f"model_final.pt"))
    
    return {
        'model_name': model_name,
        'num_parameters': num_params,
        'test_metrics': test_metrics,
        'bolivia_metrics': bolivia_metrics
    }

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('mps') if torch.backends.mps.is_available() else device
    logger.info(f'Using device: {device}')
    
    # Setup base logging directory
    args.version = f"{args.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_log_dir = f'./logs/comparison_{args.version}'
    os.makedirs(base_log_dir, exist_ok=True)
    
    # Common arguments
    args.num_classes = 2
    args.in_channels = 6
    
    # Load data (shared across all models)
    logger.info("Loading datasets...")
    train_loader = get_loader(args.data_path, DatasetType.TRAIN.value, args)
    valid_loader = get_loader(args.data_path, DatasetType.VALID.value, args)
    test_loader = get_loader(args.data_path, DatasetType.TEST.value, args)
    bolivia_loader = get_loader(args.data_path, DatasetType.BOLIVIA.value, args)
    
    # Initialize all models
    models = {
        'unet': UNet(
            in_channels=args.in_channels, 
            out_channels=args.num_classes, 
            unet_encoder_size=args.unet_out_channels
        ),
        'prithvi': PritviSegmenter(
            weights_path='./prithvi/Prithvi_100M.pt', 
            device=device, 
            output_channels=args.num_classes, 
            prithvi_encoder_size=args.prithvi_out_channels
        ),
        'prithvi_unet': PrithviUNet(
            in_channels=args.in_channels, 
            out_channels=args.num_classes, 
            weights_path='./prithvi/Prithvi_100M.pt', 
            device=device, 
            prithvi_encoder_size=args.prithvi_out_channels, 
            unet_encoder_size=args.unet_out_channels, 
            combine_method=args.combine_func, 
            dropout_prob=args.random_dropout_prob
        )
    }
    
    # Train all models and collect results
    results = []
    for model_name, model in models.items():
        model = model.to(device)
        result = train_single_model(
            model_name, model, train_loader, valid_loader, test_loader, bolivia_loader, 
            args, device, base_log_dir
        )
        results.append(result)
        
        # Clear memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Print comparison summary
    logger.info(f"\n{'='*100}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*100}")
    logger.info(f"{'Model':<20} {'Parameters':<15} {'Test IOU':<12} {'Test ACC':<12} {'Bolivia IOU':<12} {'Bolivia ACC':<12}")
    logger.info(f"{'-'*100}")
    
    for result in results:
        logger.info(
            f"{result['model_name']:<20} "
            f"{result['num_parameters']:,<15} "
            f"{result['test_metrics']['Avg_IOU']:<12.4f} "
            f"{result['test_metrics']['Avg_ACC']:<12.4f} "
            f"{result['bolivia_metrics']['Avg_IOU']:<12.4f} "
            f"{result['bolivia_metrics']['Avg_ACC']:<12.4f}"
        )
    
    # Save results to JSON
    results_file = os.path.join(base_log_dir, 'comparison_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, default=float)
    logger.info(f"\nResults saved to: {results_file}")
    
    # Find best model
    best_test_iou = max(results, key=lambda x: x['test_metrics']['Avg_IOU'])
    best_bolivia_iou = max(results, key=lambda x: x['bolivia_metrics']['Avg_IOU'])
    
    logger.info(f"\n{'='*100}")
    logger.info(f"Best model on Test set (IOU): {best_test_iou['model_name']} ({best_test_iou['test_metrics']['Avg_IOU']:.4f})")
    logger.info(f"Best model on Bolivia set (IOU): {best_bolivia_iou['model_name']} ({best_bolivia_iou['bolivia_metrics']['Avg_IOU']:.4f})")
    logger.info(f"{'='*100}\n")

if __name__ == '__main__':
    args = parse_arguments()
    main(args)