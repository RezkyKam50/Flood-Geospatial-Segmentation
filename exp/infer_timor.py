import os
import sys
import argparse
import logging
from enum import Enum
import json
import torch
import rasterio
from models.u_net import UNet
from models.prithvi_segmenter import PritviSegmenter
from models.prithvi_unet import PrithviUNet
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F
import random

class DatasetType(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    TIMOR = 'timor'
 
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

torch.manual_seed(124)

# Ground Truth Specification:
# Band 0 (Cloud/Validity Mask):
#   0 -> Ignore (no-data)
#   1 -> Valid clear land (Non-Flood)
#   2 -> Cloud
# Band 1 (Flood Annotation):
#   0 -> Ignore (no-data)
#   1 -> Non-Flood
#   2 -> Flood

USED_BANDS = (1,2,3,8,11,12)

# means and std from sen1floods11
MEANS = [0.14245495, 0.13921481, 0.12434631, 0.31420089, 0.20743526,0.12046503]
STDS = [0.04036231, 0.04186983, 0.05267646, 0.0822221 , 0.06834774, 0.05294205]

# means and std for timor (welfords online algorithm)
# MEANS = [1478.32529030, 1242.15067974, 1182.46893299, 2522.65887867, 12.16115611, 1769.97315294]
# STDS  = [749.55484295, 827.16092973, 828.88616174, 1142.21153924, 13.36843455, 967.22151682]

INPUT_SIZE = 224

def processTestIm(img, bands):
    img = img[bands, :, :].astype(np.float32)
    img = torch.tensor(img)
    norm = transforms.Normalize(MEANS, STDS)
    img = norm(img)
    return img.unsqueeze(0)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform inference on Timor flood dataset')
    parser.add_argument('--data_path', type=str, default='./datasets/Timor_ML4FLood', 
                        help='Path to the Timor data directory.')
    parser.add_argument('--output_path', type=str, default='./outputs/timor_results', 
                        help='Path to save the segmented images.')
    parser.add_argument('--bands', type=list, default=[1,2,3,8,11,12], 
                        help='Bands indices that need to be used for segmentation. Model trained on 6 bands (B, G, R, NIR, SWIR1, SWIR2).')
    parser.add_argument('--model_type', type=str, default='prithvi_unet', 
                        help='Model to use for segmentation (unet, prithvi_unet, prithvi)')
    parser.add_argument('--weights_path', type=str, default='./prithvi/Prithvi_100M.pt', 
                        help='Path to the weights file for Prithvi models.')
    parser.add_argument('--aoi', type=str, default='all', 
                        help='Which AOI to process: all, 01, 02, 03, 05, 07')
    return parser.parse_args()


def accumulate_confusion_matrix(prediction, ground_truth):
    """
    Accumulate confusion matrix values. Cloud pixels (band0 == 2) are excluded
    from metrics — only clear, valid pixels (band0 == 1, band1 != 0) are counted.
    """
    prediction = prediction.squeeze()
    ground_truth = ground_truth.squeeze()
     
    if len(ground_truth.shape) != 3 or ground_truth.shape[0] != 2:
        raise ValueError(f"Expected ground truth shape (2, H, W), got {ground_truth.shape}")
    
    band0 = ground_truth[0].astype(np.int32)  # Validity: 0=Ignore, 1=Valid, 2=Cloud
    band1 = ground_truth[1].astype(np.int32)  # Flood: 0=Ignore, 1=Non-Flood, 2=Flood
    
    # Only clear land pixels with valid annotation — clouds are excluded here
    valid_mask = (band0 == 1) & (band1 != 0)
    valid_count = np.sum(valid_mask)
    
    if valid_count == 0:
        return {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'valid_pixels': 0}
      
    gt_binary = np.zeros_like(band1)
    gt_binary[band1 == 1] = 0  
    gt_binary[band1 == 2] = 1  
    
    pred_valid = prediction[valid_mask]
    gt_valid = gt_binary[valid_mask]

    TP = np.sum((pred_valid == 1) & (gt_valid == 1))
    # FP: Predicted Flood (1), Actual Non-Flood (0)
    FP = np.sum((pred_valid == 1) & (gt_valid == 0))
    # TN: Predicted Non-Flood (0), Actual Non-Flood (0)
    TN = np.sum((pred_valid == 0) & (gt_valid == 0))
    # FN: Predicted Non-Flood (0), Actual Flood (1)
    FN = np.sum((pred_valid == 0) & (gt_valid == 1))
    
    return {
        'TP': int(TP),
        'FP': int(FP),
        'TN': int(TN),
        'FN': int(FN),
        'valid_pixels': int(valid_count)
    }


def count_prediction_pixels(prediction, ground_truth):
    """
    Count different pixel types in predictions including cloud predictions.
    Cloud pixels are NOT included in flood/non_flood counts (metrics are clean),
    but cloud_flood_pixels are counted separately for informational purposes.
    
    Returns:
        dict with counts for flood, non-flood, cloud-flood, and valid pixels
    """
    prediction = prediction.squeeze()
    ground_truth = ground_truth.squeeze()
    
    if len(ground_truth.shape) != 3 or ground_truth.shape[0] != 2:
        raise ValueError(f"Expected ground truth shape (2, H, W), got {ground_truth.shape}")
    
    band0 = ground_truth[0].astype(np.int32)  # Cloud/Validity: 0=Ignore, 1=Valid, 2=Cloud
    band1 = ground_truth[1].astype(np.int32)  # Flood: 0=Ignore, 1=Non-Flood, 2=Flood
    
    # Valid mask (clear land with annotation) — clouds excluded from metrics
    valid_mask = (band0 == 1) & (band1 != 0)
    
    # Cloud mask (informational only, not used in metrics)
    cloud_mask = (band0 == 2)
    
    # Count predictions on valid (non-cloud) pixels only
    flood_pixels = np.sum((prediction == 1) & valid_mask)
    non_flood_pixels = np.sum((prediction == 0) & valid_mask)
    
    # Count flood predictions on cloud pixels (informational, excluded from metrics)
    cloud_flood_pixels = np.sum((prediction == 1) & cloud_mask)
    
    # Total valid pixels processed (non-cloud)
    valid_pixels_processed = np.sum(valid_mask)
    
    return {
        'flood_pixels': int(flood_pixels),
        'non_flood_pixels': int(non_flood_pixels),
        'cloud_flood_pixels': int(cloud_flood_pixels),
        'valid_pixels_processed': int(valid_pixels_processed)
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


def enhance_input_for_visualization(image):
    image = image.cpu().numpy()
    image = image.squeeze()[[2, 1, 0], :, :].transpose((1, 2, 0))
    mins, maxs = np.percentile(image, PERCENTILES)
    image = (image - mins) / (maxs - mins) * 255
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


# Color constants for cloud prediction highlighting
CLOUD_COLOR_MODEL1 = (255, 165, 0)    # Orange  — model1 (the "good"  / first model) predicts flood on cloud
CLOUD_COLOR_MODEL2 = (0, 220, 220)    # Cyan    — model2 (the "bad" / second model) predicts flood on cloud


def get_2_models_visualization(good, bad, gt):
    """
    Build a comparison image for two models.

    Color scheme (priority: cloud overlays are applied last so they are always visible):
      White  (255,255,255) — both models correct (predicted flood on actual flood / non-flood on non-flood)
      Gray   (128,128,128) — no-data / ignore region
      Blue   (100,100,255) — both models wrong
      Red    (255,100,100) — model1 (good) correct only
      Green  (  0,255,  0) — model2 (bad)  correct only
      Orange (255,165,  0) — model1 (good) predicts flood on cloud pixel   [new]
      Cyan   (  0,220,220) — model2 (bad)  predicts flood on cloud pixel   [new]

    Cloud pixels are excluded from metric computation; they are only shown
    here for visual reference.
    """
    good, bad = good.squeeze(), bad.squeeze()
    gt = gt.squeeze()

    # Parse ground-truth bands
    if len(gt.shape) == 3:
        band0 = gt[0].astype(np.int32)   # 0=Ignore, 1=Valid, 2=Cloud
        band1 = gt[1].astype(np.int32)   # 0=Ignore, 1=Non-Flood, 2=Flood
        gt_binary = np.zeros_like(band1)
        gt_binary[band1 == 2] = 1
        no_data = (band0 != 1) | (band1 == 0)
        cloud_mask = (band0 == 2)
    else:
        gt_binary = gt
        no_data = (gt == -1)
        cloud_mask = np.zeros_like(gt, dtype=bool)

    output = np.zeros((good.shape[0], good.shape[1], 3), dtype=np.uint8)

    # --- Base layers (non-cloud valid pixels) ---
    # Default: white where either model predicts flood
    output[good == 1] = [255, 255, 255]
    output[bad == 1]  = [255, 255, 255]

    # No-data / ignore (gray) — applied over defaults
    output[no_data] = [128, 128, 128]

    # Both models wrong — blue
    both_wrong = (
        ((good == 0) & (bad == 0) & (gt_binary == 1)) |
        ((good == 1) & (bad == 1) & (gt_binary == 0))
    )
    output[both_wrong & ~no_data] = [100, 100, 255]

    # model1 (good) correct only — red
    good_only = (
        ((good == 0) & (bad == 1) & (gt_binary == 1)) |
        ((good == 1) & (bad == 0) & (gt_binary == 0))
    )
    output[good_only & ~no_data] = [255, 100, 100]

    # model2 (bad) correct only — green
    bad_only = (
        ((good == 1) & (bad == 0) & (gt_binary == 1)) |
        ((good == 0) & (bad == 1) & (gt_binary == 0))
    )
    output[bad_only & ~no_data] = [0, 255, 0]

    # --- Cloud overlay (applied last so always visible) ---
    # Orange: model1 predicts flood on a cloud pixel
    output[(good == 1) & cloud_mask] = list(CLOUD_COLOR_MODEL1)
    # Cyan: model2 predicts flood on a cloud pixel
    output[(bad == 1) & cloud_mask] = list(CLOUD_COLOR_MODEL2)

    return output


def create_legend(comparison_title, model1_name, model2_name, height=280, width=480):
    """
    Create a legend panel that includes the two new cloud-prediction color entries
    (orange for model1, cyan for model2).
    """
    legend = np.ones((height, width, 3), dtype=np.uint8) * 250
    legend_img = Image.fromarray(legend)
    draw = ImageDraw.Draw(legend_img)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_text  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except:
        try:
            font_title = ImageFont.truetype("arial.ttf", 16)
            font_text  = ImageFont.truetype("arial.ttf", 13)
        except:
            font_title = ImageFont.load_default()
            font_text  = font_title

    y_offset = 15
    draw.text((width // 2, y_offset), comparison_title,
              fill=(0, 0, 0), font=font_title, anchor="mt")
    y_offset += 35

    legend_entries = [
        ((255, 255, 255), "Both models correct (White)"),
        ((128, 128, 128), "No data / cloud region (Gray)"),
        ((100, 100, 255), "Both models wrong (Blue)"),
        ((255, 100, 100), f"{model1_name} correct only (Red)"),
        ((0,  255,   0),  f"{model2_name} correct only (Green)"),
        # New cloud-prediction entries
        (CLOUD_COLOR_MODEL1, f"{model1_name} flood pred on cloud (Orange)"),
        (CLOUD_COLOR_MODEL2, f"{model2_name} flood pred on cloud (Cyan)"),
    ]

    box_size = 22
    for color, description in legend_entries:
        draw.rectangle(
            [20, y_offset, 20 + box_size, y_offset + box_size],
            fill=color, outline=(0, 0, 0), width=2
        )
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
        font_title    = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_subtitle = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        try:
            font_title    = ImageFont.truetype("arial.ttf", 18)
            font_subtitle = ImageFont.truetype("arial.ttf", 12)
        except:
            font_title    = ImageFont.load_default()
            font_subtitle = font_title

    y_offset = 15
    draw.text((width // 2, y_offset), lines[0],
              fill=(0, 0, 0), font=font_title, anchor="mt")
    y_offset += 30

    for line in lines[1:]:
        draw.text((width // 2, y_offset), line,
                  fill=(50, 50, 50), font=font_subtitle, anchor="mt")
        y_offset += 20

    return np.array(label_img)


def add_blue_gap(images, gap_size=10):
    total_width = sum(img.shape[1] for img in images) + (len(images) - 1) * gap_size
    max_height  = max(img.shape[0] for img in images)
    result = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255
    result[:, :, 0:2] = 0

    current_x = 0
    for img in images:
        result[:img.shape[0], current_x:current_x + img.shape[1]] = img
        current_x += img.shape[1] + gap_size

    return result


def stack_vertically(images, gap_size=5):
    total_height = sum(img.shape[0] for img in images) + (len(images) - 1) * gap_size
    max_width    = max(img.shape[1] for img in images)
    result = np.ones((total_height, max_width, 3), dtype=np.uint8) * 255

    current_y = 0
    for img in images:
        x_offset = (max_width - img.shape[1]) // 2
        result[current_y:current_y + img.shape[0], x_offset:x_offset + img.shape[1]] = img
        current_y += img.shape[0] + gap_size

    return result


def load_model(model_type, model_path, device, weights_path):
    if model_type == 'unet':
        model = UNet(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES)
    elif model_type == 'prithvi_unet':
        model = PrithviUNet(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES,
                            weights_path=weights_path, device=device)
    elif model_type == 'prithvi':
        model = PritviSegmenter(output_channels=NUM_CLASSES, weights_path=weights_path, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device)
    return model


def segment_image_with_sliding_window(model, image, window_size, stride, device):
    model.eval()
    _, _, h_img, w_img = image.shape
    h_stride, w_stride = stride
    h_crop,   w_crop   = window_size
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    preds     = torch.zeros((1, NUM_CLASSES, h_img, w_img), device=device)
    count_mat = torch.zeros((1, 1,           h_img, w_img), device=device)

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

            preds[:, :, y1:y2, x1:x2]     += crop_seg
            count_mat[:, :, y1:y2, x1:x2] += 1

    preds = preds / count_mat
    preds = torch.argmax(preds, dim=1)
    return preds


def get_gt(data_root):
    ground_truth = f"{data_root}/GT"

    timor_data_dir = [
        f"{ground_truth}/EMSR507_AOI01_DEL_PRODUCT",   # Pleiades-1A-1B
        f"{ground_truth}/EMSR507_AOI02_DEL_PRODUCT",   # PlanetScope
        f"{ground_truth}/EMSR507_AOI03_DEL_PRODUCT",   # PlanetScope
        f"{ground_truth}/EMSR507_AOI05_DEL_PRODUCT",   # Sentinel-2
        f"{ground_truth}/EMSR507_AOI07_GRA_PRODUCT"    # PlanetScope
    ]

    def read_include_file(filepath):
        if not os.path.exists(filepath):
            logger.warning(f"Include file not found: {filepath}")
            return []
        with open(filepath, 'r') as file:
            return [line.strip() for line in file if line.strip()]

    def get_files_for_aoi(dir_path):
        incl_file = f"{dir_path}/include.txt"
        aoi_name  = os.path.basename(dir_path)
        indices   = read_include_file(incl_file)
        return [f"{dir_path}/{aoi_name}_{i}.tif" for i in indices]

    return {
        "tif01_gt": get_files_for_aoi(timor_data_dir[0]),
        "tif02_gt": get_files_for_aoi(timor_data_dir[1]),
        "tif03_gt": get_files_for_aoi(timor_data_dir[2]),
        "tif05_gt": get_files_for_aoi(timor_data_dir[3]),
        "tif07_gt": get_files_for_aoi(timor_data_dir[4])
    }


def get_s2(data_root):
    """Get Sentinel-2 input files"""
    s2_path = f"{data_root}/S2"

    timor_data_dir = [
        f"{s2_path}/EMSR507_AOI01_DEL_PRODUCT",  
        f"{s2_path}/EMSR507_AOI02_DEL_PRODUCT",
        f"{s2_path}/EMSR507_AOI03_DEL_PRODUCT", 
        f"{s2_path}/EMSR507_AOI05_DEL_PRODUCT",
        f"{s2_path}/EMSR507_AOI07_GRA_PRODUCT"
    ]

    def read_include_file(filepath):
        if not os.path.exists(filepath):
            logger.warning(f"Include file not found: {filepath}")
            return []
        with open(filepath, 'r') as file:
            return [line.strip() for line in file if line.strip()]

    def get_files_for_aoi(dir_path):
        incl_file = f"{dir_path}/include.txt"
        aoi_name  = os.path.basename(dir_path)
        indices   = read_include_file(incl_file)
        return [f"{dir_path}/{aoi_name}_{i}.tif" for i in indices]

    return {
        "tif01_s2": get_files_for_aoi(timor_data_dir[0]),
        "tif02_s2": get_files_for_aoi(timor_data_dir[1]),
        "tif03_s2": get_files_for_aoi(timor_data_dir[2]),
        "tif05_s2": get_files_for_aoi(timor_data_dir[3]),
        "tif07_s2": get_files_for_aoi(timor_data_dir[4])
    }


def main():
    
    args = parse_arguments()
    os.makedirs(args.output_path, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('mps') if torch.backends.mps.is_available() else device
    logger.info(f'Using device: {device}')

    s2_data = get_s2(args.data_path)
    gt_data = get_gt(args.data_path)
     
    if args.aoi != 'all':
        aoi_key = f'tif{args.aoi}_s2'
        gt_key  = f'tif{args.aoi}_gt'
        if aoi_key in s2_data and gt_key in gt_data:
            s2_data = {aoi_key: s2_data[aoi_key]}
            gt_data = {gt_key:  gt_data[gt_key]}
        else:
            logger.error(f"AOI {args.aoi} not found in data")
            return
     
    total_samples = sum(len(files) for files in s2_data.values())
    logger.info(f'Loaded {total_samples} samples for processing')
     
    models     = {}
    model_names = ['unet', 'prithvi_unet', 'prithvi']
    for model_type in model_names:
        logger.info(f'Loading model: {model_type}')
        models[model_type] = load_model(model_type, models_paths[model_type],
                                        device, args.weights_path)

    # Cumulative confusion matrices for overall (aggregated) metrics
    # Cloud pixels are excluded from all metric accumulation (see accumulate_confusion_matrix)
    confusion_matrices = {mt: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for mt in model_names}

    # Per-image metrics lists for standard deviation computation
    per_image_metrics = {mt: {
        'IOU_floods': [],
        'IOU_non_floods': [],
        'Avg_IOU': [],
        'ACC_floods': [],
        'ACC_non_floods': [],
        'Avg_ACC': []
    } for mt in model_names}

    pixel_counts = {
        mt: {
            'total_flood_pixels': 0,
            'total_non_flood_pixels': 0,
            'total_cloud_flood_pixels': 0,
            'total_valid_pixels_processed': 0
        } for mt in model_names
    }
     
    for aoi_key in s2_data.keys():
        aoi_num = aoi_key.split('_')[0].replace('tif', '')
        gt_key  = f'tif{aoi_num}_gt'
        
        logger.info(f'\nProcessing AOI {aoi_num}')
        
        s2_files = s2_data[aoi_key]
        gt_files = gt_data[gt_key]
        
        if len(s2_files) != len(gt_files):
            logger.warning(f'Mismatch in file counts for AOI {aoi_num}: '
                           f'{len(s2_files)} S2 files vs {len(gt_files)} GT files')
         
        for idx, (s2_file, gt_file) in enumerate(zip(s2_files, gt_files)):
            logger.info(f'Processing file {idx+1}/{len(s2_files)} in AOI {aoi_num}')
            
            try:
                with rasterio.open(s2_file) as src:
                    image = src.read()
                with rasterio.open(gt_file) as src:
                    mask = src.read()
                
                logger.info(f'Image shape: {image.shape}, Mask shape: {mask.shape}')
                 
                if mask.shape[0] != 2:
                    logger.error(f'Ground truth must have 2 bands, got {mask.shape[0]}')
                    continue
                 
                band0_unique = np.unique(mask[0])
                band1_unique = np.unique(mask[1])
                logger.info(f'Band 0 (Cloud/Valid) unique values: {band0_unique}')
                logger.info(f'Band 1 (Flood) unique values: {band1_unique}')
                 
                image_tensor = processTestIm(image, args.bands).to(device)
                 
                window_size = (224, 224)
                stride      = (128, 128)
                predictions = {}
                
                for model_type, model in models.items():
                    segmented_image = segment_image_with_sliding_window(
                        model, image_tensor, window_size, stride, device)
                    predictions[model_type] = segmented_image.cpu().numpy()

                    # Metrics — cloud pixels are excluded inside accumulate_confusion_matrix
                    cm = accumulate_confusion_matrix(predictions[model_type], mask)
                    confusion_matrices[model_type]['TP'] += cm['TP']
                    confusion_matrices[model_type]['FP'] += cm['FP']
                    confusion_matrices[model_type]['TN'] += cm['TN']
                    confusion_matrices[model_type]['FN'] += cm['FN']

                    img_metrics = calculate_metrics_from_confusion(cm)
                    for metric_name, value in img_metrics.items():
                        per_image_metrics[model_type][metric_name].append(value)

                    # Pixel counts — cloud pixels tracked separately, not in metrics
                    pc = count_prediction_pixels(predictions[model_type], mask)
                    pixel_counts[model_type]['total_flood_pixels']         += pc['flood_pixels']
                    pixel_counts[model_type]['total_non_flood_pixels']     += pc['non_flood_pixels']
                    pixel_counts[model_type]['total_cloud_flood_pixels']   += pc['cloud_flood_pixels']
                    pixel_counts[model_type]['total_valid_pixels_processed'] += pc['valid_pixels_processed']
                    
                    logger.debug(f'{model_type} - TP:{cm["TP"]} FP:{cm["FP"]} TN:{cm["TN"]} FN:{cm["FN"]} '
                                 f'Valid:{cm["valid_pixels"]}')
                 
                input_vis   = enhance_input_for_visualization(image_tensor)
                input_label = create_input_label(f"RGB Input Image\nAOI {aoi_num}",
                                                 input_vis.shape[1])
                input_with_label = stack_vertically([input_label, input_vis], gap_size=0)

                # --- Comparison visualisations ---
                # In each call: arg1 = model1 (orange on cloud), arg2 = model2 (cyan on cloud)
                uprithvi_unet    = get_2_models_visualization(predictions['prithvi_unet'],
                                                              predictions['unet'], mask)
                uprithvi_prithvi = get_2_models_visualization(predictions['prithvi'],
                                                              predictions['prithvi_unet'], mask)
                prithvi_unet     = get_2_models_visualization(predictions['prithvi'],
                                                              predictions['unet'], mask)

                # Legends — model1 = orange on cloud, model2 = cyan on cloud
                legend1 = create_legend("Prithvi-UNet vs UNet",       "Prithvi-UNet", "UNet")
                legend2 = create_legend("Prithvi vs Prithvi-UNet",    "Prithvi",      "Prithvi-UNet")
                legend3 = create_legend("Prithvi vs UNet",            "Prithvi",      "UNet")
                 
                comp1_with_legend = stack_vertically([legend1, uprithvi_unet],    gap_size=5)
                comp2_with_legend = stack_vertically([legend2, uprithvi_prithvi], gap_size=5)
                comp3_with_legend = stack_vertically([legend3, prithvi_unet],     gap_size=5)
                 
                final_image = add_blue_gap(
                    [input_with_label, comp1_with_legend, comp2_with_legend, comp3_with_legend],
                    gap_size=15
                )
                 
                output_filename = os.path.basename(s2_file).replace('.tif', '_comparison.png')
                output_path     = os.path.join(args.output_path, f'AOI{aoi_num}_{output_filename}')
                
                final_image = Image.fromarray(final_image)
                final_image.save(output_path)
                logger.info(f'Saved visualization to {output_path}')
                
            except Exception as e:
                logger.error(f'Error processing {s2_file}: {str(e)}')
                import traceback
                traceback.print_exc()
                continue
     
    logger.info('\n' + '='*80)
    logger.info('CUMULATIVE PIXEL COUNTS (After All Inference)')
    logger.info('Note: flood/non-flood counts are on valid (non-cloud) pixels only.')
    logger.info('      cloud_flood_pixels are informational — excluded from all metrics.')
    logger.info('='*80)
    
    for model_type in model_names:
        pc = pixel_counts[model_type]
        logger.info(f'\n{model_type.upper()} Model:')
        logger.info(f'  Total Flood Pixels Predicted (valid only):  {pc["total_flood_pixels"]:,}')
        logger.info(f'  Total Non-Flood Pixels Predicted (valid):   {pc["total_non_flood_pixels"]:,}')
        logger.info(f'  Total Cloud-Flood Pixels (excl. from mtr):  {pc["total_cloud_flood_pixels"]:,}')
        logger.info(f'  Total Valid Pixels Processed (non-cloud):   {pc["total_valid_pixels_processed"]:,}')
        
        if pc['total_valid_pixels_processed'] > 0:
            flood_pct     = (pc['total_flood_pixels']     / pc['total_valid_pixels_processed']) * 100
            non_flood_pct = (pc['total_non_flood_pixels'] / pc['total_valid_pixels_processed']) * 100
            logger.info(f'  Flood Percentage (of valid):               {flood_pct:.2f}%')
            logger.info(f'  Non-Flood Percentage (of valid):           {non_flood_pct:.2f}%')
    
    logger.info('\n' + '='*80)
    logger.info('OVERALL mIoU RESULTS  (cloud pixels excluded from all metrics)')
    logger.info('='*80)
     
    results = {}
    
    for model_type in model_names:
        metrics = calculate_metrics_from_confusion(confusion_matrices[model_type])

        std_metrics = {
            metric_name: float(np.std(values))
            for metric_name, values in per_image_metrics[model_type].items()
        }

        logger.info(f'{model_type:15} - Mean mIoU: {metrics["Avg_IOU"]:.4f} ± {std_metrics["Avg_IOU"]:.4f} '
                    f'(Flood: {metrics["IOU_floods"]:.4f} ± {std_metrics["IOU_floods"]:.4f}, '
                    f'Non-flood: {metrics["IOU_non_floods"]:.4f} ± {std_metrics["IOU_non_floods"]:.4f}) '
                    f'Accuracy: {metrics["Avg_ACC"]:.4f} ± {std_metrics["Avg_ACC"]:.4f}')
         
        results[model_type] = {
            'test_metrics': {
                'IOU_floods':     metrics['IOU_floods'],
                'IOU_non_floods': metrics['IOU_non_floods'],
                'Avg_IOU':        metrics['Avg_IOU'],
                'ACC_floods':     metrics['ACC_floods'],
                'ACC_non_floods': metrics['ACC_non_floods'],
                'Avg_ACC':        metrics['Avg_ACC']
            },
            'std_metrics':    std_metrics,
            'confusion_matrix': confusion_matrices[model_type],
            'pixel_counts':   pixel_counts[model_type]
        }
     
    json_output_path = os.path.join(args.output_path, 'metrics_results.json')
    with open(json_output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f'\nMetrics saved to: {json_output_path}')
    logger.info('='*80)


if __name__ == '__main__':
    main()