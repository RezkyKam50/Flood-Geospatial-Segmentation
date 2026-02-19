import os
import sys
import argparse
import logging
from enum import Enum
import json
import torch
from data_loading.sen1floods11 import processTestIm
import rasterio
from models.u_net import UNet
from models.prithvi_segmenter import PritviSegmenter
from models.prithvi_unet import PrithviUNet
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from collections import defaultdict


class DatasetType(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    BOLIVIA = 'bolivia'
 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

NUM_CLASSES = 2
IN_CHANNELS = 6

models_paths = {
    'unet': 'logs/Bolivia_100E_TVERSKY_A0.3_B0.7_G2.0_E100_seed_12/unet/models/model_final.pt',
    'prithvi': 'logs/Bolivia_100E_TVERSKY_A0.3_B0.7_G2.0_E100_seed_124/prithvi/models/model_final.pt',
    'prithvi_unet': './logs/Bolivia_100E_TVERSKY_A0.3_B0.7_G0.75_E100_seed_48/prithvi_unet/models/model_final.pt',
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a Segmentation model')
    parser.add_argument('--data_path', type=str, default='./datasets/sen1floods11_v1.1', help='Path to the data directory.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test', 'bolivia'], help='Dataset split to use (train, valid, or test).')
    parser.add_argument('--output_path', type=str, help='Path to save the segmented image.')
    parser.add_argument('--bands', type=list, default=[1,2,3,8,11,12], help='Bands indices that need to be used for segmentation. Recall that the model was trained on 6 bands (B, G, R, NIR, SWIR1, SWIR2).')
    parser.add_argument('--model_type', type=str, default='prithvi_unet', help='Model to use for segmentation (unet, prithvi_unet, prithvi)')
    parser.add_argument('--weights_path', type=str, default='./prithvi/Prithvi_100M.pt', help='Path to the weights file for Prithvi models.')
    return parser.parse_args()
 
args = parse_arguments()

def segment_image_with_sliding_window(model, image, window_size, stride, device):
    model.eval()   
    _, _, h_img, w_img = image.shape   
    h_stride, w_stride = stride  
    h_crop, w_crop = window_size  
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
 
    preds = torch.zeros((1, NUM_CLASSES, h_img, w_img), device=device)
    count_mat = torch.zeros((1, 1, h_img, w_img), device=device)
 
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
  
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
 
            crop_img = image[:, :, y1:y2, x1:x2]
            with torch.no_grad():   
                crop_seg_logit = model(crop_img)   
                crop_seg = torch.softmax(crop_seg_logit, dim=1)   
 
            preds[:, :, y1:y2, x1:x2] += crop_seg
            count_mat[:, :, y1:y2, x1:x2] += 1
 
    preds = preds / count_mat
    preds = torch.argmax(preds, dim=1)  
    return preds

def load_model(model_type, model_path, device, weights_path):
    if model_type == 'unet':
        model = UNet(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES)
    elif model_type == 'prithvi_unet':
        model = PrithviUNet(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES, weights_path=weights_path, device=device)
    elif model_type == 'prithvi':
        model = PritviSegmenter(output_channels=NUM_CLASSES, weights_path=weights_path, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device)
    return model


def accumulate_confusion_matrix(prediction, ground_truth, ignore_index=255):
 
    prediction = prediction.squeeze()
    ground_truth = ground_truth.squeeze()
    
    # Create mask for valid pixels (exclude ignore_index)
    valid_mask = (ground_truth != ignore_index) & (ground_truth != -1)
    
    # Apply mask to both prediction and ground truth
    pred_valid = prediction[valid_mask]
    gt_valid = ground_truth[valid_mask]
    
    # Calculate confusion matrix components
    # For binary segmentation: class 1 = flood, class 0 = non-flood
    TP = np.sum((pred_valid == 1) & (gt_valid == 1))  # True Positives (flood correctly identified)
    FP = np.sum((pred_valid == 1) & (gt_valid == 0))  # False Positives (non-flood predicted as flood)
    TN = np.sum((pred_valid == 0) & (gt_valid == 0))  # True Negatives (non-flood correctly identified)
    FN = np.sum((pred_valid == 0) & (gt_valid == 1))  # False Negatives (flood predicted as non-flood)
    
    return {
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN
    }


def calculate_metrics_from_confusion(metrics_dict):
 
    TP = metrics_dict['TP']
    FP = metrics_dict['FP']
    TN = metrics_dict['TN']
    FN = metrics_dict['FN']
     
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
        'Avg_ACC': Avg_ACC
    }

PERCENTILES = (0.1, 99.9)
NO_DATA_FLOAT = 255

def enhance_mask_for_visualization(mask, no_data_pixel, gt):
    gt = gt.squeeze()
    mask = mask.squeeze() * 255
    print(mask.shape)
    mask[no_data_pixel] = 128
    # have 3 channels for visualization
    
    
    mask_extended = np.stack([mask, mask, mask], axis=2)
    
    # pixels that were wrongly classified label as red
    mask_extended[((mask == 0) & (gt == 1)) | ((mask == 255) & (gt == 0))] = [255, 0, 0]
    return mask_extended.astype(np.uint8)

def get_2_models_visualization(good, bad, gt):
    good, bad = good.squeeze(), bad.squeeze()
    gt = gt.squeeze()
    
    output = np.zeros((good.shape[0], good.shape[1], 3), dtype=np.uint8)
    output[good == 1] = [255, 255, 255]
    output[bad == 1] = [255, 255, 255]
    output[gt == -1] = [128, 128, 128]
    # both models were wrong = blue
    output[((good == 0) & (bad == 0) & (gt == 1)) | ((good == 1) & (bad == 1) & (gt == 0))] = [100, 100, 255]
    # unet was correct, prithvi was wrong = red
    output[((good == 0) & (bad == 1) & (gt == 1)) | ((good == 1) & (bad == 0) & (gt == 0))] = [255, 100, 100]
    # prithvi was correct, unet was wrong = green
    output[((good == 1) & (bad == 0) & (gt == 1)) | ((good == 0) & (bad == 1) & (gt == 0))] = [0, 255, 0]
    return output
    

def enhance_input_for_visualization(image):
    image = image.cpu().numpy()
    image = image.squeeze()[[2, 1, 0], :, :].transpose((1, 2, 0))
    mins, maxs = np.percentile(image, PERCENTILES)
 
    image = (image - mins) / (maxs - mins) * 255
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def create_legend(comparison_title, model1_name, model2_name, height=220, width=450):
    legend = np.ones((height, width, 3), dtype=np.uint8) * 250  # Light gray background
    legend_img = Image.fromarray(legend)
    draw = ImageDraw.Draw(legend_img)
     
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except:
        try:
            font_title = ImageFont.truetype("arial.ttf", 16)
            font_text = ImageFont.truetype("arial.ttf", 13)
        except:
            font_title = ImageFont.load_default()
            font_text = font_title
    
    y_offset = 15
     
    draw.text((width//2, y_offset), comparison_title, fill=(0, 0, 0), font=font_title, anchor="mt")
    y_offset += 35
     
    legend_entries = [
        ((255, 255, 255), "Both models correct (White)"),
        ((128, 128, 128), "No data available (Gray)"),
        ((100, 100, 255), "Both models wrong (Blue)"),
        ((255, 100, 100), f"{model1_name} correct only (Red)"),
        ((0, 255, 0), f"{model2_name} correct only (Green)"),
    ]
    
    box_size = 22
    for color, description in legend_entries:
 
        draw.rectangle([20, y_offset, 20 + box_size, y_offset + box_size], 
                      fill=color, outline=(0, 0, 0), width=2)
 
        draw.text((50, y_offset + 2), description, fill=(0, 0, 0), font=font_text)
        y_offset += 32
    
    return np.array(legend_img)


def create_input_label(text, width, height=None):
 
    lines = text.split('\n')
     
    if height is None:
        base_height = 40
        line_height = 20
        height = base_height + (len(lines) - 1) * line_height
    
    label = np.ones((height, width, 3), dtype=np.uint8) * 250
    label_img = Image.fromarray(label)
    draw = ImageDraw.Draw(label_img)
    
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_subtitle = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        try:
            font_title = ImageFont.truetype("arial.ttf", 18)
            font_subtitle = ImageFont.truetype("arial.ttf", 12)
        except:
            font_title = ImageFont.load_default()
            font_subtitle = font_title
     
    y_offset = 15
    draw.text((width//2, y_offset), lines[0], fill=(0, 0, 0), font=font_title, anchor="mt")
    y_offset += 30
     
    for line in lines[1:]:
        draw.text((width//2, y_offset), line, fill=(50, 50, 50), font=font_subtitle, anchor="mt")
        y_offset += 20
    
    return np.array(label_img)
    

def add_blue_gap(images, gap_size=10):
 
    total_width = sum(img.shape[1] for img in images) + (len(images) - 1) * gap_size
    max_height = max(img.shape[0] for img in images)
    result = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255   
    result[:, :, 0:2] = 0   

    current_x = 0
    for img in images:
        result[:img.shape[0], current_x:current_x + img.shape[1]] = img
        current_x += img.shape[1] + gap_size

    return result


def stack_vertically(images, gap_size=5):
 
    total_height = sum(img.shape[0] for img in images) + (len(images) - 1) * gap_size
    max_width = max(img.shape[1] for img in images)
    result = np.ones((total_height, max_width, 3), dtype=np.uint8) * 255
    
    current_y = 0
    for img in images:
 
        x_offset = (max_width - img.shape[1]) // 2
        result[current_y:current_y + img.shape[0], x_offset:x_offset + img.shape[1]] = img
        current_y += img.shape[0] + gap_size
    
    return result


def load_data_from_txt(data_path, split):
 
    txt_path = os.path.join(data_path, 'splits', f'flood_{split}_data.txt')
    data_files = []
    with open(txt_path, 'r') as f:
        for line in f:
            identifier = line.strip()
            if identifier:   
                img_path = f"{identifier}_S2Hand.tif"
                mask_path = f"{identifier}_LabelHand.tif"
                data_files.append((img_path, mask_path))
    return data_files

def main():
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('mps') if torch.backends.mps.is_available() else device
    logger.info(f'Using device: {device}')

    data_files = load_data_from_txt(args.data_path, args.split)
    logger.info(f'Loaded {len(data_files)} samples from {args.split} split')
    
    models = {}
    model_names = ['unet', 'prithvi_unet', 'prithvi']
    for model_type in model_names:
        args.model_type = model_type
        models[model_type] = load_model(args.model_type, models_paths[model_type], device, args.weights_path)

    # Cumulative confusion matrices for overall (aggregated) metrics
    confusion_matrices = {model_type: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for model_type in model_names}

    # Per-image metrics lists for standard deviation computation
    per_image_metrics = {model_type: {
        'IOU_floods': [],
        'IOU_non_floods': [],
        'Avg_IOU': [],
        'ACC_floods': [],
        'ACC_non_floods': [],
        'Avg_ACC': []
    } for model_type in model_names}
    
    for filename, mask_filename in data_files:
        image = rasterio.open(os.path.join(args.data_path, 'data/S2L1CHand', filename)).read()
        mask = rasterio.open(os.path.join(args.data_path, 'data/LabelHand', mask_filename)).read()
        image = processTestIm(image, args.bands).to(device)
     
        window_size = (224, 224)
        stride = (128, 128)

        predictions = {}
 
        for model_type, model in models.items():
            segmented_image = segment_image_with_sliding_window(model, image, window_size, stride, device)
            predictions[model_type] = segmented_image.cpu().numpy()
             
            cm = accumulate_confusion_matrix(predictions[model_type], mask)
            confusion_matrices[model_type]['TP'] += cm['TP']
            confusion_matrices[model_type]['FP'] += cm['FP']
            confusion_matrices[model_type]['TN'] += cm['TN']
            confusion_matrices[model_type]['FN'] += cm['FN']

            # Compute per-image metrics and store for std dev
            img_metrics = calculate_metrics_from_confusion(cm)
            for metric_name, value in img_metrics.items():
                per_image_metrics[model_type][metric_name].append(value)

        print(mask.max(), mask.min())
         
        input_to_stored = enhance_input_for_visualization(image)
         
        band_mapping = {
            1: 'Coastal Aerosol',
            2: 'Blue',
            3: 'Green',
            4: 'Red',
            5: 'Red Edge 1',
            6: 'Red Edge 2',
            7: 'Red Edge 3',
            8: 'NIR',
            9: 'Water Vapor',
            10: 'SWIR-Cirrus',
            11: 'SWIR1',
            12: 'SWIR2'
        }

        bands_list = [band_mapping.get(i, f'Band {i}') for i in args.bands]
        bands_text = '\n'.join(bands_list)
        input_label = create_input_label(f"RGB Input Image\n{bands_text}", input_to_stored.shape[1])
        input_with_label = stack_vertically([input_label, input_to_stored], gap_size=0)
         
        uprithvi_unet = get_2_models_visualization(predictions['prithvi_unet'], predictions['unet'], mask)
        uprithvi_prithvi = get_2_models_visualization(predictions['prithvi_unet'], predictions['prithvi'], mask)
        prithvi_unet = get_2_models_visualization(predictions['prithvi'], predictions['unet'], mask)
         
        legend1 = create_legend("Prithvi-UNet vs UNet", "Prithvi-UNet", "UNet")
        legend2 = create_legend("Prithvi-UNet vs Prithvi", "Prithvi-UNet", "Prithvi")
        legend3 = create_legend("Prithvi vs UNet", "Prithvi", "UNet")
         
        comp1_with_legend = stack_vertically([legend1, uprithvi_unet], gap_size=5)
        comp2_with_legend = stack_vertically([legend2, uprithvi_prithvi], gap_size=5)
        comp3_with_legend = stack_vertically([legend3, prithvi_unet], gap_size=5)
         
        final_image = add_blue_gap([input_with_label, comp1_with_legend, comp2_with_legend, comp3_with_legend], gap_size=15)

        final_image = Image.fromarray(final_image)
        final_image.save(os.path.join(args.output_path, f'{filename}.png'))
        logger.info(f'Saved visualization to {os.path.join(args.output_path, f"{filename}.png")}')

    logger.info('\n' + '=' * 60)
    logger.info('OVERALL mIoU RESULTS')
    logger.info('=' * 60)

    results = {}

    for model_type in model_names:
        metrics = calculate_metrics_from_confusion(confusion_matrices[model_type])

        # Compute standard deviation across per-image metrics
        std_metrics = {
            metric_name: float(np.std(values))
            for metric_name, values in per_image_metrics[model_type].items()
        }

        logger.info(
            f"{model_type:15} - Mean mIoU: {metrics['Avg_IOU']:.4f} ± {std_metrics['Avg_IOU']:.4f} "
            f"(Flood: {metrics['IOU_floods']:.4f} ± {std_metrics['IOU_floods']:.4f}, "
            f"Non-flood: {metrics['IOU_non_floods']:.4f} ± {std_metrics['IOU_non_floods']:.4f})"
        )

        results[model_type] = {
            'test_metrics': {
                'IOU_floods': metrics['IOU_floods'],
                'IOU_non_floods': metrics['IOU_non_floods'],
                'Avg_IOU': metrics['Avg_IOU'],
                'ACC_floods': metrics['ACC_floods'],
                'ACC_non_floods': metrics['ACC_non_floods'],
                'Avg_ACC': metrics['Avg_ACC']
            },
            'std_metrics': std_metrics,
            'confusion_matrix': {k: int(v) for k, v in confusion_matrices[model_type].items()}
        }

    logger.info("=" * 60)

    # Save all metrics (mean + std) for all 3 models to JSON
    json_output_path = os.path.join(args.output_path, 'metrics_results.json')
    os.makedirs(args.output_path, exist_ok=True)
    with open(json_output_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f'Metrics saved to: {json_output_path}')

if __name__ == '__main__':
    main()