from utils.testing import computeIOU, computeAccuracy, computeMetrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from data_loading.sen1floods11 import get_loader
import argparse

from utils.customloss import DiceLoss, DiceLoss2 
from segmentation_models_pytorch.losses import FocalLoss, LovaszLoss, JaccardLoss, TverskyLoss

from models.u_net import UNet
from models.prithvi_segmenter import PritviSegmenter
from models.prithvi_unet import PrithviUNet

from loguru import logger
import os
import json

class DatasetType(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    BOLIVIA = 'bolivia'


def parse_arguments():
    parser = argparse.ArgumentParser(description='Compare Prithvi, UNet, and Prithvi-UNet models')
    parser.add_argument('--data_path', type=str, default='./datasets/sen1floods11_v1.1', help='Path to the data directory.')
    parser.add_argument('--version', type=str, default='comparison', help='Experiment version')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--loss_func', type=str, default='bce', help='Loss function to use: bce, dice, dice2, focal, lovasz, tversky')
    parser.add_argument('--prithvi_out_channels', type=int, default=768, help='Number of output channels from the Prithvi encoders')
    parser.add_argument('--unet_out_channels', type=int, default=768, help='Number of output channels from the UNet encoders')
    parser.add_argument('--prithvi_finetune_ratio', type=float, default=1, help='Fine-tune ratio for Prithvi models')
    parser.add_argument('--save_model_interval', type=int, default=5, help='Save the model every n epochs')
    parser.add_argument('--test_interval', type=int, default=1, help='Test the model every n epochs')
    parser.add_argument('--combine_func', type=str, default='concat', choices=['concat', 'mul', 'add'], help='Combination function for U-Prithvi')
    parser.add_argument('--random_dropout_prob', type=float, default=2/3, help='Dropout probability for U-Prithvi')
    
    return parser.parse_args()


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


def construct_mets(model_name, model, bolivia_metrics):

    num_train_params = get_number_of_trainable_parameters(model)
    total_params = get_total_parameters(model)

    metrics = {
        'model_name': model_name,
        'trainable_parameters': num_train_params,
        'total_parameters': total_params, 
        'bolivia_metrics': bolivia_metrics
    }

    return metrics

# parameter yg dapat update gradien
def get_number_of_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# total semua parameter (update gradien + frozen params)
def get_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bolivia_loader = get_loader(args.data_path, DatasetType.BOLIVIA.value, args)

    if args.loss_func == 'diceloss':
        criterion = DiceLoss(device=device)
    elif args.loss_func == 'dl2':
        criterion = DiceLoss2(device=device, epsilon=1e-7)
    elif args.loss_func == 'bce':
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.7, 0.3], device=device), ignore_index=255)
    elif args.loss_func == 'focal':
        criterion = FocalLoss(mode="multiclass", alpha=0.25, gamma=2, ignore_index=255, reduction='mean')
    elif args.loss_func == 'lovasz':
        criterion = LovaszLoss(mode='multiclass', per_image=False, from_logits=True, ignore_index=255)
    elif args.loss_func == 'tversky':
        criterion = TverskyLoss(mode='multiclass', alpha=0.3, beta=0.7, gamma=1.33, eps=1e-7, ignore_index=255, from_logits=True)

    args.num_classes = 2
    args.in_channels = 6

    models = {
        'prithvi': PritviSegmenter(
            weights_path='./prithvi/Prithvi-EO-1.0-100M-sen1floods11/sen1floods11_Prithvi_100M.pth', 
            device=device, 
            output_channels=args.num_classes, 
            prithvi_encoder_size=args.prithvi_out_channels
        )
    }

    results = []
    for model_name, model in models.items():
        model = model.to(device)
        bolivia_metrics = test(model, bolivia_loader, criterion, device)
        results.append(construct_mets(model_name, model, bolivia_metrics))

    logger.info(f"Bolivia Set - Avg IOU: {bolivia_metrics['Avg_IOU']:.4f}, Avg ACC: {bolivia_metrics['Avg_ACC']:.4f}, Loss: {bolivia_metrics['Loss']:.4f}")
 
    os.makedirs('IOU_Metrics', exist_ok=True)

    results_file = os.path.join('IOU_Metrics', 'results_bolivia.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, default=float)
    logger.info(f"\nResults saved to: {results_file}")

