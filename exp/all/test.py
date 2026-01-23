import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
from models.u_net import UNet
from models.prithvi_segmenter import PritviSegmenter
from models.prithvi_unet import PrithviUNet
from data_loading.ml4floods import timor_leste_events, test_path_s2, test_path_label, extension, getArrFlood, USED_BANDS, MEANS, STDS, INPUT_SIZE, PATCH_SIZE, processTimorLesteData
from torchvision import transforms

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize model comparisons on Timor-Leste')
    parser.add_argument('--data_path', type=str, default='./datasets/sen1floods11_v1.1', 
                        help='Path to the data directory.')
    parser.add_argument('--models_dir', type=str, required=True,
                        help='Directory containing trained models (e.g., ./logs/comparison_xxx)')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Batch size for inference (reduce if OOM)')
    parser.add_argument('--output_path', type=str, default='./comparison_timor_leste.png',
                        help='Path to save the comparison figure')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of patches to visualize (default: all)')
    parser.add_argument('--samples_per_event', type=int, default=3,
                        help='Number of patches to sample from each event')
    
    # Model parameters (should match training)
    parser.add_argument('--prithvi_out_channels', type=int, default=768)
    parser.add_argument('--unet_out_channels', type=int, default=768)
    parser.add_argument('--combine_func', type=str, default='concat')
    parser.add_argument('--random_dropout_prob', type=float, default=2/3)
    
    return parser.parse_args()

def load_model(model_name, args, device):
    """Load a trained model"""
    args.num_classes = 2
    args.in_channels = 6
    
    if model_name == 'unet':
        model = UNet(
            in_channels=args.in_channels,
            out_channels=args.num_classes,
            unet_encoder_size=args.unet_out_channels
        )
    elif model_name == 'prithvi':
        model = PritviSegmenter(
            weights_path='./prithvi/Prithvi_100M.pt',
            device=device,
            output_channels=args.num_classes,
            prithvi_encoder_size=args.prithvi_out_channels
        )
    elif model_name == 'prithvi_unet':
        model = PrithviUNet(
            in_channels=args.in_channels,
            out_channels=args.num_classes,
            weights_path='./prithvi/Prithvi_100M.pt',
            device=device,
            prithvi_encoder_size=args.prithvi_out_channels,
            unet_encoder_size=args.unet_out_channels,
            combine_method=args.combine_func,
            dropout_prob=args.random_dropout_prob
        )
    
    # Load weights
    model_path = os.path.join(args.models_dir, model_name, 'models', 'model_final.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model

def load_timor_leste_sampled_patches(samples_per_event=3):
    """
    Load a subset of patches from each Timor-Leste event.
    This avoids loading all patches into memory at once.
    """
    all_samples = []
    
    for event_id, satellite in timor_leste_events.items():
        img_path = f"{test_path_s2}{event_id}{extension}"
        label_path = f"{test_path_label}{event_id}{extension}"
        
        if not os.path.exists(img_path) or not os.path.exists(label_path):
            raise ValueError(f"File not found: {img_path} or {label_path}")
        
        # Load raw arrays
        arr_x = np.nan_to_num(getArrFlood(img_path))
        arr_y = getArrFlood(label_path)
        
        # Convert ml4floods labels to binary
        arr_y_new = np.zeros_like(arr_y)
        arr_y_new[arr_y == 0] = 255  # invalid -> ignore
        arr_y_new[arr_y == 1] = 0    # land -> land
        arr_y_new[arr_y == 2] = 1    # water -> water
        arr_y_new[arr_y == 3] = 1    # permanent water -> water
        
        # Get all patches for this event
        patches = processTimorLesteData((arr_x, arr_y_new))
        
        # Sample evenly spaced patches
        total_patches = len(patches)
        if samples_per_event >= total_patches:
            selected_indices = range(total_patches)
        else:
            # Evenly space samples across all patches
            step = total_patches / samples_per_event
            selected_indices = [int(i * step) for i in range(samples_per_event)]
        
        # Create samples for selected patches
        for idx in selected_indices:
            img, label = patches[idx]
            all_samples.append({
                'image': img,
                'label': label,
                'event_id': event_id,
                'satellite': satellite,
                'patch_idx': idx,
                'total_patches': total_patches
            })
    
    return all_samples

def get_predictions_batch(models, images, device, batch_size=8):
    """
    Get predictions from all models in batches to avoid OOM.
    
    Args:
        models: Dictionary of {model_name: model}
        images: Tensor of images [N, C, H, W]
        device: torch device
        batch_size: Batch size for inference
    
    Returns:
        Dictionary of {model_name: predictions_array}
    """
    num_samples = images.shape[0]
    predictions = {name: [] for name in models.keys()}
    
    # Process in batches
    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch_imgs = images[i:batch_end].to(device)
        
        with torch.no_grad():
            for model_name, model in models.items():
                outputs = model(batch_imgs)
                pred = torch.argmax(outputs, dim=1)
                predictions[model_name].append(pred.cpu())
        
        # Clear GPU memory after each batch
        del batch_imgs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Concatenate all batch predictions
    for model_name in predictions:
        predictions[model_name] = torch.cat(predictions[model_name], dim=0).numpy()
    
    return predictions

def create_comparison_figure(models, samples, device, save_path='comparison.png', batch_size=8):
    """
    Create a comparison figure showing RGB, Ground Truth, and predictions from all models
    """
    num_samples = len(samples)
    
    # Prepare all images (keep on CPU initially)
    imgs = torch.stack([s['image'] for s in samples])
    masks = torch.stack([s['label'] for s in samples])
    
    # Get predictions in batches
    print(f"  Running inference in batches of {batch_size}...")
    predictions = get_predictions_batch(models, imgs, device, batch_size)
    
    # Convert to numpy
    imgs_np = imgs.numpy()
    masks_np = masks.numpy()
    
    # Create figure
    num_cols = 2 + len(models)
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(4*num_cols, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    model_names = list(models.keys())
    
    for i in range(num_samples):
        col = 0
        
        # Column 0: RGB visualization
        rgb = imgs_np[i, :3, :, :].transpose(1, 2, 0)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        axes[i, col].imshow(rgb)
        if i == 0:
            axes[i, col].set_title('Input RGB', fontsize=14, fontweight='bold')
        axes[i, col].axis('off')
        col += 1
        
        # Column 1: Ground Truth
        gt_vis = np.ma.masked_where(masks_np[i] == 255, masks_np[i])
        axes[i, col].imshow(gt_vis, cmap='Blues', vmin=0, vmax=1)
        if i == 0:
            axes[i, col].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[i, col].axis('off')
        col += 1
        
        # Remaining columns: Model predictions
        for model_name in model_names:
            axes[i, col].imshow(predictions[model_name][i], cmap='Blues', vmin=0, vmax=1)
            if i == 0:
                display_name = model_name.replace('_', '-').upper()
                if display_name == 'PRITHVI-UNET':
                    display_name = 'U-Prithvi'
                axes[i, col].set_title(f'{display_name}', fontsize=14, fontweight='bold')
            axes[i, col].axis('off')
            col += 1
        
        # Add sample label
        event_id = samples[i]['event_id']
        patch_idx = samples[i]['patch_idx']
        total_patches = samples[i]['total_patches']
        axes[i, 0].text(-0.1, 0.5, f'{event_id}\nPatch {patch_idx}/{total_patches}', 
                       transform=axes[i, 0].transAxes,
                       fontsize=9, fontweight='bold',
                       verticalalignment='center',
                       horizontalalignment='right',
                       rotation=90)
    
    plt.suptitle(f'Model Comparison on Timor-Leste Test Set ({num_samples} Patches from 5 Events)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison figure saved to: {save_path}")
    plt.close(fig)

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('mps') if torch.backends.mps.is_available() else device
    print(f'Using device: {device}')
    
    # Load sampled Timor-Leste patches
    print(f"Loading {args.samples_per_event} patches per event from Timor-Leste...")
    samples = load_timor_leste_sampled_patches(samples_per_event=args.samples_per_event)
    print(f"Loaded {len(samples)} total patches from 5 events:")
    
    # Count patches per event
    from collections import Counter
    event_counts = Counter([s['event_id'] for s in samples])
    for event_id, count in event_counts.items():
        satellite = samples[[s['event_id'] for s in samples].index(event_id)]['satellite']
        total = samples[[s['event_id'] for s in samples].index(event_id)]['total_patches']
        print(f"  - {event_id} ({satellite}): {count}/{total} patches")
    
    # Optionally limit total number of samples
    if args.max_samples and len(samples) > args.max_samples:
        print(f"\nLimiting to {args.max_samples} samples for visualization...")
        samples = samples[:args.max_samples]
    
    # Load all models
    print("\nLoading models...")
    models = {}
    for model_name in ['unet', 'prithvi', 'prithvi_unet']:
        try:
            print(f"  Loading {model_name}...")
            models[model_name] = load_model(model_name, args, device)
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            continue
    
    if not models:
        raise ValueError("No models could be loaded. Check your models_dir path.")
    
    print(f"Successfully loaded {len(models)} models: {list(models.keys())}")
    
    # Create comparison figure
    print(f"\nCreating comparison figure with {len(samples)} patches...")
    create_comparison_figure(
        models, 
        samples,
        device, 
        save_path=args.output_path,
        batch_size=args.batch_size
    )
    
    print("Done!")

if __name__ == '__main__':
    args = parse_arguments()
    main(args)