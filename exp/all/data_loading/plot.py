import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import random
from time import time
import csv
import os
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import rasterio

class InMemoryDataset(torch.utils.data.Dataset):
  def __init__(self, data_list, preprocess_func):
    self.data_list = data_list
    self.preprocess_func = preprocess_func
  
  def __getitem__(self, i):
    return self.preprocess_func(self.data_list[i])
  
  def __len__(self):
    return len(self.data_list)

'''
Pleiades-1A-1B:
Panchromatic: 480-830 nm
Blue: 430-550 nm
Green: 490-610 nm
Red: 600-720 nm
NIR: 750-950 nm

PlanetScope:
Coastal Blue: 431 - 452 nm
Blue: 465 – 515 nm
Green I: 513 - 549 nm
Green: 547 – 583 nm
Yellow: 600 - 620 nm
Red: 650 – 680 nm
RedEdge: 697 – 713 nm
NIR: 845 – 885 nm

Sentinel-2:
Coastal Aerosol: 433 - 453 nm
Blue: 458 - 523 nm
Green: 543 - 578 nm
Red: 650 - 680 nm
Vegetation Red Edge 1: 698 - 713 nm
Vegetation Red Edge 2: 733 - 748 nm
Vegetation Red Edge 3: 773 - 793 nm
NIR: 785 - 900 nm
Narrow NIR: 855 - 875 nm
Water Vapour: 935 - 955 nm
Cirrus: 1360 - 1390 nm
SWIR 1: 1565 - 1655 nm
SWIR 2: 2100 - 2280 nm
'''

# for S2 bands
SATELLITE_ALL_BANDS_MAPPING = {
    'Sentinel-2': {
        'bands': list(range(1, 14)),  # All 13 bands (1-13)
        'names': ['Coastal Aerosol', 'Blue', 'Green', 'Red', 
                 'Red Edge 1', 'Red Edge 2', 'Red Edge 3', 'NIR', 
                 'Narrow NIR', 'Water Vapour', 'Cirrus', 'SWIR1', 'SWIR2']
    },
    'Pleiades-1A-1B': {
        'bands': list(range(1, 6)),  # All 5 bands (1-5): Pan, Blue, Green, Red, NIR
        'names': ['Panchromatic', 'Blue', 'Green', 'Red', 'NIR']
    },
    'PlanetScope': {
        'bands': list(range(1, 9)),  # All 8 bands (1-8)
        'names': ['Coastal Blue', 'Blue', 'Green I', 'Green', 
                 'Yellow', 'Red', 'RedEdge', 'NIR']
    }
}
 
SATELLITE_BAND_MAPPING = {
    'Sentinel-2': (1, 2, 3, 8, 11, 12),  # Blue, Green, Red, NIR, SWIR1, SWIR2
    'Pleiades-1A-1B': (1, 2, 3, 4, None, None),  # Blue, Green, Red, NIR, no SWIR
    'PlanetScope': (2, 4, 6, 8, None, None),  # Blue, Green, Red, NIR, no SWIR
}

INPUT_SIZE = 224
SENTINEL_MEANS = [0.13692222, 0.13376727, 0.11943894, 0.30450596, 0.20170933, 0.11685023]
SENTINEL_STDS = [0.03381057, 0.03535441, 0.04496607, 0.07556641, 0.06130259, 0.04689224]

root = 'datasets/WorldFloodsv2'
metadata_path = f'{root}/dataset_metadata.csv'
train_path = f'{root}/train/PERMANENTWATERJRC/'

extension = '.tif'
timor_leste_events = {
    "EMSR507_AOI01_DEL_PRODUCT": "Pleiades-1A-1B",
    "EMSR507_AOI02_DEL_PRODUCT": "PlanetScope",
    "EMSR507_AOI03_DEL_PRODUCT": "PlanetScope",
    "EMSR507_AOI05_DEL_PRODUCT": "Sentinel-2",
    "EMSR507_AOI07_GRA_PRODUCT": "PlanetScope"
}

files = [(f"{train_path}{event_id}{extension}", satellite) 
         for event_id, satellite in timor_leste_events.items()]


def get_all_bands_for_satellite(file_path, satellite_type):
    band_config = SATELLITE_ALL_BANDS_MAPPING[satellite_type]
    band_indices = band_config['bands']
    
    with rasterio.open(file_path) as src:
        actual_band_count = src.count
        bands_data = []
         
        for band_idx in band_indices:
            if band_idx <= actual_band_count:
                bands_data.append(src.read(band_idx).astype(np.float32))
            else:
                # If band doesn't exist, create a zero array
                bands_data.append(np.zeros((src.height, src.width), dtype=np.float32))
                print(f"Warning: {file_path} - Band {band_idx} not found (file has {actual_band_count} bands), using zeros")
        
        return np.stack(bands_data, axis=0)


def get_bands_for_satellite(file_path, satellite_type):
    band_indices = SATELLITE_BAND_MAPPING[satellite_type]
    
    with rasterio.open(file_path) as src:
        bands_data = []
        for band_idx in band_indices:
            if band_idx is None:
                bands_data.append(np.zeros((src.height, src.width), dtype=np.float32))
            else:
                bands_data.append(src.read(band_idx).astype(np.float32))
        
        return np.stack(bands_data, axis=0)


def plot_tif(file_satellite_pairs: list, bands_to_display=(0, 1, 2), use_all_bands=False):
    num_files = len(file_satellite_pairs)
    with sns.axes_style("white"):
        fig, axes = plt.subplots(num_files, 2, figsize=(15, 5 * num_files))
        fig.patch.set_facecolor('white')
        if num_files == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (file_path, satellite_type) in enumerate(file_satellite_pairs):
            if use_all_bands:
                img_array = get_all_bands_for_satellite(file_path, satellite_type)
                band_config = SATELLITE_ALL_BANDS_MAPPING[satellite_type]
                band_names = band_config['names']
                num_bands = len(band_config['bands'])
            else:
                img_array = get_bands_for_satellite(file_path, satellite_type)
                band_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
                num_bands = 6
             
            rgb_img = np.stack([img_array[i] for i in bands_to_display[:3]], axis=-1)
            rgb_normalized = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
            
            axes[idx, 0].imshow(rgb_normalized)
            axes[idx, 0].set_title(
                f"{os.path.basename(file_path)}\n({satellite_type})", 
                fontsize=12, 
                fontweight='bold',
                pad=10,
                color='#2c3e50'
            )
            axes[idx, 0].axis('off')
            
            for spine in axes[idx, 0].spines.values():
                spine.set_edgecolor('#ecf0f1')
                spine.set_linewidth(2)
            
            with rasterio.open(file_path) as src:
                if use_all_bands:
                    bands_info = '\n  '.join([f"{name}: Band {idx}" 
                                             for name, idx in zip(band_names, band_config['bands'])])
                else:
                    band_mapping = SATELLITE_BAND_MAPPING[satellite_type]
                    bands_info = '\n  '.join([f"{name}: Band {idx if idx else 'N/A'}" 
                                             for name, idx in zip(band_names, band_mapping)])
                
                info = rf"""
$\bf{{File:}}$ {os.path.basename(file_path)}
$\bf{{Satellite:}}$ {satellite_type}
$\bf{{Shape:}}$ {src.height} × {src.width} px
$\bf{{Loaded\ bands:}}$ {num_bands} / {src.count} total
$\bf{{Band\ mapping:}}$
  {bands_info}
$\bf{{RGB\ display:}}$ {list(bands_to_display[:3])}
$\bf{{Value\ range:}}$ {img_array.min():.2f} to {img_array.max():.2f}
$\bf{{Mean:}}$ {img_array.mean():.4f}
$\bf{{Std:}}$ {img_array.std():.4f}
$\bf{{CRS:}}$ {src.crs}
                """
            
            axes[idx, 1].text(
                0.05, 0.5, info,
                fontfamily='monospace',
                fontsize=11,
                verticalalignment='center',
                transform=axes[idx, 1].transAxes,
                bbox=dict(
                    boxstyle='round,pad=1',
                    facecolor='#f8f9fa',
                    edgecolor='#dee2e6',
                    linewidth=1.5,
                    alpha=0.8
                ),
                color='#2c3e50'
            )
            axes[idx, 1].axis('off')
        
        plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=3.0)
        os.makedirs('./exp/plots', exist_ok=True)
        suffix = '_all_bands' if use_all_bands else ''
        plt.savefig(f'./exp/plots/timor_leste_samples{suffix}.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved to ./exp/plots/timor_leste_samples{suffix}.png")


def plot_all_bands(file_path, satellite_type):
    band_config = SATELLITE_ALL_BANDS_MAPPING[satellite_type]
    band_indices = band_config['bands']
    band_names = band_config['names']
    num_bands = len(band_indices)
    
    with rasterio.open(file_path) as src:
        ncols = min(4, num_bands)  
        nrows = (num_bands + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        fig.suptitle(f"{os.path.basename(file_path)} - {satellite_type}\n {num_bands} Bands", 
                     fontsize=14, fontweight='bold')
         
        if num_bands == 1:
            axes = np.array([axes])
        axes = axes.flatten() if num_bands > 1 else axes
        
        for plot_idx, (band_idx, band_name) in enumerate(zip(band_indices, band_names)):
            band_data = src.read(band_idx)
            normalized = (band_data - band_data.min()) / (band_data.max() - band_data.min() + 1e-8)
            axes[plot_idx].imshow(normalized, cmap='gray')
            axes[plot_idx].set_title(
                f'{band_name} (Band {band_idx})\nRange: [{band_data.min():.2f}, {band_data.max():.2f}]',
                fontsize=10
            )
            axes[plot_idx].axis('off')
         
        for i in range(num_bands, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout(pad=2.0, h_pad=5.0, w_pad=3.0)
        os.makedirs('./exp/plots', exist_ok=True)
        safe_name = os.path.basename(file_path).replace('.tif', '')
        plt.savefig(f'./exp/plots/{safe_name}_all_bands.png', dpi=150, bbox_inches='tight')
        print(f"Saved: ./exp/plots/{safe_name}_all_bands.png")

 
# plot_tif(files, use_all_bands=False)
plot_tif(files, use_all_bands=True)
  
for file_path, satellite_type in files:
    plot_all_bands(file_path, satellite_type) 