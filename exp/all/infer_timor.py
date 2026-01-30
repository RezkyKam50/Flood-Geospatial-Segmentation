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
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F
import random

class DatasetType(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    BOLIVIA = 'bolivia'
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
    'unet': 'logs/bolivia_100E_TVERSKY/unet/models/model_final.pt',
    'prithvi': 'logs/bolivia_100E_TVERSKY/prithvi_sen1floods/models/model_final.pt',
    'prithvi_unet': 'logs/bolivia_100E_TVERSKY/prithvi_unet/models/model_final.pt',
}

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
MEANS = [0.14245495, 0.13921481, 0.12434631, 0.31420089, 0.20743526,0.12046503]
STDS = [0.04036231, 0.04186983, 0.05267646, 0.0822221 , 0.06834774, 0.05294205]
INPUT_SIZE = 224

def processTestIm(img, bands):
    img = img[bands, :, :].astype(np.float32)
    img = torch.tensor(img)
    norm = transforms.Normalize(MEANS, STDS)
    img = norm(img)
    return img.unsqueeze(0)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform inference on Timor flood dataset')
    parser.add_argument('--data_path', type=str, default='./datasets/Timor_ML4Floods', 
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
 
    prediction = prediction.squeeze()
    ground_truth = ground_truth.squeeze()
     
    if len(ground_truth.shape) != 3 or ground_truth.shape[0] != 2:
        raise ValueError(f"Expected ground truth shape (2, H, W), got {ground_truth.shape}")
    
    band0 = ground_truth[0].astype(np.int32)  # Validity: 0=Ignore, 1=Valid, 2=Cloud
    band1 = ground_truth[1].astype(np.int32)  # Flood: 0=Ignore, 1=Non-Flood, 2=Flood
    
    #  must be clear (Band0==1) and have annotation (Band1!=0)
    valid_mask = (band0 == 1) & (band1 != 0)
    valid_count = np.sum(valid_mask)
    
    if valid_count == 0:
        return {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'valid_pixels': 0}
     
    # Band1: 1->0 (Non-Flood), 2->1 (Flood)
    gt_binary = np.zeros_like(band1)
    gt_binary[band1 == 1] = 0  # Non-Flood
    gt_binary[band1 == 2] = 1  # Flood
    
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
        aoi_name = os.path.basename(dir_path)
        indices = read_include_file(incl_file)
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
        aoi_name = os.path.basename(dir_path)
        indices = read_include_file(incl_file)
        return [f"{dir_path}/{aoi_name}_{i}.tif" for i in indices]

    return {
        "tif01_s2": get_files_for_aoi(timor_data_dir[0]),
        "tif02_s2": get_files_for_aoi(timor_data_dir[1]),
        "tif03_s2": get_files_for_aoi(timor_data_dir[2]),
        "tif05_s2": get_files_for_aoi(timor_data_dir[3]),
        "tif07_s2": get_files_for_aoi(timor_data_dir[4])
    }


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
        model = PrithviUNet(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES, 
                           weights_path=weights_path, device=device)
    elif model_type == 'prithvi':
        model = PritviSegmenter(output_channels=NUM_CLASSES, 
                               weights_path=weights_path, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device)
    return model


PERCENTILES = (0.1, 99.9)

def enhance_input_for_visualization(image):
    image = image.cpu().numpy()
    image = image.squeeze()[[2, 1, 0], :, :].transpose((1, 2, 0))
    mins, maxs = np.percentile(image, PERCENTILES)
    image = (image - mins) / (maxs - mins) * 255
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def get_2_models_visualization(pred1, pred2, gt):
 
    pred1 = pred1.squeeze()
    pred2 = pred2.squeeze()
    gt = gt.squeeze()
    
    if len(gt.shape) != 3 or gt.shape[0] != 2:
        raise ValueError(f"Expected GT shape (2, H, W), got {gt.shape}")
    
    band0 = gt[0].astype(np.int32)  # Cloud/Validity: 0=Ignore, 1=Valid, 2=Cloud
    band1 = gt[1].astype(np.int32)  # Flood: 0=Ignore, 1=Non-Flood, 2=Flood
    
    h, w = pred1.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Create binary ground truth map: -1=NoData, 0=Non-Flood, 1=Flood
    gt_binary = np.full_like(band1, -1, dtype=np.int32)
    gt_binary[band1 == 1] = 0  # Non-Flood
    gt_binary[band1 == 2] = 1  # Flood
 
    gt_binary[(band1 == 0) | (band0 != 1)] = -1
     
    cloud_mask = (band0 == 2)
 
    pred1_cloud_flood = cloud_mask & (pred1 == 1)
    pred2_cloud_flood = cloud_mask & (pred2 == 1)
     
    both_cloud_flood = pred1_cloud_flood & pred2_cloud_flood
    output[both_cloud_flood] = [255, 0, 255]  # Magenta RGB
     
    only_pred1_cloud = pred1_cloud_flood & (~pred2_cloud_flood)
    output[only_pred1_cloud] = [255, 255, 0]  # Yellow RGB
     
    only_pred2_cloud = (~pred1_cloud_flood) & pred2_cloud_flood
    output[only_pred2_cloud] = [0, 255, 255]  # Cyan RGB
     
    any_cloud_flood = pred1_cloud_flood | pred2_cloud_flood
    nodata_mask = (gt_binary == -1) & ~any_cloud_flood
    
    # No data (Black) - only where not already colored as cloud prediction
    output[nodata_mask] = [0, 0, 0]
    
    # Masks for valid pixels (non-cloud, non-nodata)
    valid_mask = gt_binary != -1
    nonflood_mask = gt_binary == 0
    flood_mask = gt_binary == 1
     
    p1_correct = pred1 == gt_binary
    p2_correct = pred2 == gt_binary
     
    both_correct_land = p1_correct & p2_correct & nonflood_mask
    output[both_correct_land] = [64, 64, 64]   
    
    both_correct_flood = p1_correct & p2_correct & flood_mask
    output[both_correct_flood] = [100, 100, 255]   
     
    both_wrong = (~p1_correct) & (~p2_correct) & valid_mask
    output[both_wrong] = [255, 255, 255]   
     
    only_pred1 = p1_correct & (~p2_correct) & valid_mask
    output[only_pred1] = [255, 100, 100]
     
    only_pred2 = (~p1_correct) & p2_correct & valid_mask
    output[only_pred2] = [0, 255, 0]
    
    return output


def create_legend(comparison_title, model1_name, model2_name, height=380, width=450):
    """Create legend image with separate entries for each model's cloud predictions."""
    legend = np.ones((height, width, 3), dtype=np.uint8) * 250
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
        ((64, 64, 64), "Both correct on land (Gray)"),
        ((100, 100, 255), "Both model is wrong (Blue)"),
        ((255, 0, 255), "Both prediction on Cloud (Magenta)"),
        ((255, 255, 0), f"{model1_name} prediction on Cloud (Yellow)"),
        ((0, 255, 255), f"{model2_name} prediction on Cloud (Cyan)"),
        ((0, 0, 0), "No data (Black)"),
        ((255, 255, 255), "Both model is correct (White)"),
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
        gt_key = f'tif{args.aoi}_gt'
        if aoi_key in s2_data and gt_key in gt_data:
            s2_data = {aoi_key: s2_data[aoi_key]}
            gt_data = {gt_key: gt_data[gt_key]}
        else:
            logger.error(f"AOI {args.aoi} not found in data")
            return
     
    total_samples = sum(len(files) for files in s2_data.values())
    logger.info(f'Loaded {total_samples} samples for processing')
     
    models = {}
    model_names = ['unet', 'prithvi_unet', 'prithvi']
    for model_type in model_names:
        logger.info(f'Loading model: {model_type}')
        models[model_type] = load_model(model_type, models_paths[model_type], 
                                       device, args.weights_path)
     
    confusion_matrices = {model_type: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for model_type in model_names}
     
    for aoi_key in s2_data.keys():
        aoi_num = aoi_key.split('_')[0].replace('tif', '')
        gt_key = f'tif{aoi_num}_gt'
        
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
                stride = (128, 128)
                predictions = {}
                
                for model_type, model in models.items():
                    segmented_image = segment_image_with_sliding_window(
                        model, image_tensor, window_size, stride, device)
                    predictions[model_type] = segmented_image.cpu().numpy()
                     
                    cm = accumulate_confusion_matrix(predictions[model_type], mask)
                    confusion_matrices[model_type]['TP'] += cm['TP']
                    confusion_matrices[model_type]['FP'] += cm['FP']
                    confusion_matrices[model_type]['TN'] += cm['TN']
                    confusion_matrices[model_type]['FN'] += cm['FN']
                    
                    logger.debug(f'{model_type} - TP:{cm["TP"]} FP:{cm["FP"]} TN:{cm["TN"]} FN:{cm["FN"]} '
                                f'Valid:{cm["valid_pixels"]}')
                 
                input_vis = enhance_input_for_visualization(image_tensor)
                 
                input_label = create_input_label(f"RGB Input Image\nAOI {aoi_num}", 
                                                 input_vis.shape[1])
                input_with_label = stack_vertically([input_label, input_vis], gap_size=0)
                  
                uprithvi_unet = get_2_models_visualization(predictions['prithvi_unet'], 
                                                          predictions['unet'], mask)
                 
                uprithvi_prithvi = get_2_models_visualization(predictions['prithvi'], 
                                                             predictions['prithvi_unet'], mask)
                 
                prithvi_unet = get_2_models_visualization(predictions['prithvi'], 
                                                         predictions['unet'], mask)
                 
                legend1 = create_legend("Prithvi-UNet vs UNet", "Prithvi-UNet", "UNet")
                legend2 = create_legend("Prithvi vs Prithvi-UNet", "Prithvi", "Prithvi-UNet")
                legend3 = create_legend("Prithvi vs UNet", "Prithvi", "UNet")
                 
                comp1_with_legend = stack_vertically([legend1, uprithvi_unet], gap_size=5)
                comp2_with_legend = stack_vertically([legend2, uprithvi_prithvi], gap_size=5)
                comp3_with_legend = stack_vertically([legend3, prithvi_unet], gap_size=5)
                 
                final_image = add_blue_gap([input_with_label, comp1_with_legend, 
                                           comp2_with_legend, comp3_with_legend], gap_size=15)
                 
                output_filename = os.path.basename(s2_file).replace('.tif', '_comparison.png')
                output_path = os.path.join(args.output_path, f'AOI{aoi_num}_{output_filename}')
                
                final_image = Image.fromarray(final_image)
                final_image.save(output_path)
                logger.info(f'Saved visualization to {output_path}')
                
            except Exception as e:
                logger.error(f'Error processing {s2_file}: {str(e)}')
                import traceback
                traceback.print_exc()
                continue
     
    logger.info('\n' + '='*60)
    logger.info('OVERALL mIoU RESULTS')
    logger.info('='*60)
     
    results = {}
    
    for model_type in model_names:
        metrics = calculate_metrics_from_confusion(confusion_matrices[model_type])
        logger.info(f'{model_type:15} - Mean mIoU: {metrics["Avg_IOU"]:.4f} '
                   f'(Flood: {metrics["IOU_floods"]:.4f}, Non-flood: {metrics["IOU_non_floods"]:.4f}) '
                   f'Accuracy: {metrics["Avg_ACC"]:.4f}')
         
        results[model_type] = {
            'test_metrics': {
                'IOU_floods': metrics['IOU_floods'],
                'IOU_non_floods': metrics['IOU_non_floods'],
                'Avg_IOU': metrics['Avg_IOU'],
                'ACC_floods': metrics['ACC_floods'],
                'ACC_non_floods': metrics['ACC_non_floods'],
                'Avg_ACC': metrics['Avg_ACC']
            },
            'confusion_matrix': confusion_matrices[model_type]
        }
     
    json_output_path = os.path.join(args.output_path, 'metrics_results.json')
    with open(json_output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f'\nMetrics saved to: {json_output_path}')
    logger.info('='*60)


if __name__ == '__main__':
    main()