import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import random
from time import time
import csv
import os
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import geopandas as gpd
from rasterio import features
from rasterio.transform import from_bounds
from matplotlib.patches import Patch

class InMemoryDataset(torch.utils.data.Dataset):
  def __init__(self, data_list, preprocess_func):
    self.data_list = data_list
    self.preprocess_func = preprocess_func
  
  def __getitem__(self, i):
    return self.preprocess_func(self.data_list[i])
  
  def __len__(self):
    return len(self.data_list)

SATELLITE_ALL_BANDS_MAPPING = {
    'Sentinel-2': {
        'bands': list(range(1, 14)),
        'names': ['Coastal Aerosol', 'Blue', 'Green', 'Red', 
                 'Red Edge 1', 'Red Edge 2', 'Red Edge 3', 'NIR', 
                 'Narrow NIR', 'Water Vapour', 'Cirrus', 'SWIR1', 'SWIR2']
    },
    'Pleiades-1A-1B': {
        'bands': list(range(1, 6)),
        'names': ['Panchromatic', 'Blue', 'Green', 'Red', 'NIR']
    },
    'PlanetScope': {
        'bands': list(range(1, 9)),
        'names': ['Coastal Blue', 'Blue', 'Green I', 'Green', 
                 'Yellow', 'Red', 'RedEdge', 'NIR']
    }
}

INPUT_SIZE = 224
SENTINEL_MEANS = [0.13692222, 0.13376727, 0.11943894, 0.30450596, 0.20170933, 0.11685023]
SENTINEL_STDS = [0.03381057, 0.03535441, 0.04496607, 0.07556641, 0.06130259, 0.04689224]

root = 'datasets/WorldFloodsv2'
metadata_path = f'{root}/dataset_metadata.csv'
train_path = f'{root}/train/S2/' # supports all bands
train_watermask_path = f'{root}/train/gt/' # supports only 1 bands

geojson_path = f'{root}/train/floodmaps/'

geo_extension = '.geojson'
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

files_geojson = [(f"{geojson_path}{event_id}{geo_extension}", satellite) 
         for event_id, satellite in timor_leste_events.items()]

files_watermask = [(f"{train_watermask_path}{event_id}{extension}", satellite) 
         for event_id, satellite in timor_leste_events.items()]

def getArrFlood(fname):
  return rasterio.open(fname).read()


# ========== DATA LOADING FUNCTIONS ==========

def get_bands_for_satellite(file_path, satellite_type):
    """Load satellite bands from TIF file based on satellite type."""
    band_indices = SATELLITE_ALL_BANDS_MAPPING[satellite_type]['bands']
    
    with rasterio.open(file_path) as src:
        bands_data = []
        for band_idx in band_indices:
            if band_idx is None:
                bands_data.append(np.zeros((src.height, src.width), dtype=np.float32))
            else:
                bands_data.append(src.read(band_idx).astype(np.float32))
        
        return np.stack(bands_data, axis=0)


def load_satellite_data(tif_path, satellite_type, bands_to_display=(0, 1, 2)):
    """Load and prepare satellite imagery data."""
    img_array = get_bands_for_satellite(tif_path, satellite_type)
    
    # Create RGB composite
    rgb_img = np.stack([img_array[i] for i in bands_to_display[:3]], axis=-1)
    rgb_normalized = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
    
    # Get metadata
    with rasterio.open(tif_path) as src:
        crs = src.crs
        transform = src.transform
        height = src.height
        width = src.width
    
    return {
        'rgb_normalized': rgb_normalized,
        'img_array': img_array,
        'crs': crs,
        'transform': transform,
        'height': height,
        'width': width
    }


def load_geojson_mask(geojson_path, height, width, transform, crs):
    if not os.path.exists(geojson_path):
        return None, None
    
    gdf = gpd.read_file(geojson_path)
    
    # Reproject if needed
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    
    # Create rasterized mask
    mask = features.rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    
    return mask, gdf


def load_permanent_water_mask(watermask_path):
    """Load permanent water mask from TIF file."""
    if not os.path.exists(watermask_path):
        return None
    
    with rasterio.open(watermask_path) as src:
        # Read first band
        water_mask = src.read(1)
        
        # Convert to binary (assuming 1=land, 2=water, 3=permanent water based on ml4floods)
        # Or could be different - adjust based on actual data
        binary_mask = np.zeros_like(water_mask, dtype=np.uint8)
        binary_mask[water_mask == 2] = 1  # water
        binary_mask[water_mask == 3] = 1  # permanent water
        
        return binary_mask


# ========== PLOTTING FUNCTIONS ==========

def plot_tif_with_geojson_and_water(tif_geojson_water_pairs, bands_to_display=(0, 1, 2)):
    """Plot multiple TIF files with their GeoJSON masks and permanent water in a grid."""
    num_files = len(tif_geojson_water_pairs)
    
    with sns.axes_style("white"):
        fig, axes = plt.subplots(num_files, 4, figsize=(24, 6 * num_files))
        fig.patch.set_facecolor('white')
        
        if num_files == 1:
            axes = axes.reshape(1, -1)
        
        for idx, ((tif_path, geojson_path, watermask_path), satellite_type) in enumerate(tif_geojson_water_pairs):
            # Load satellite data
            sat_data = load_satellite_data(tif_path, satellite_type, bands_to_display)
            rgb_normalized = sat_data['rgb_normalized']
            
            # Plot 1: Original RGB
            axes[idx, 0].imshow(rgb_normalized)
            axes[idx, 0].set_title(
                f"RGB Image\n{os.path.basename(tif_path)}\n({satellite_type})", 
                fontsize=11, fontweight='bold', color='#2c3e50'
            )
            axes[idx, 0].axis('off')
            
            # Load permanent water mask
            water_mask = load_permanent_water_mask(watermask_path)
            
            # Plot 2: Permanent Water Mask
            if water_mask is not None:
                axes[idx, 1].imshow(water_mask, cmap='Blues', interpolation='nearest')
                water_pixel_count = water_mask.sum()
                axes[idx, 1].set_title(
                    f"Permanent Water\n{water_pixel_count:,} pixels", 
                    fontsize=11, fontweight='bold', color='#2c3e50'
                )
                axes[idx, 1].axis('off')
            else:
                axes[idx, 1].text(0.5, 0.5, f"Water mask not found", 
                                 ha='center', va='center', fontsize=10, color='red')
                axes[idx, 1].axis('off')
            
            # Load GeoJSON flood mask
            flood_mask, gdf = load_geojson_mask(
                geojson_path, 
                sat_data['height'], 
                sat_data['width'], 
                sat_data['transform'], 
                sat_data['crs']
            )
            
            # Plot 3: Flood mask
            if flood_mask is not None:
                axes[idx, 2].imshow(flood_mask, cmap='Reds', interpolation='nearest')
                axes[idx, 2].set_title(
                    f"Flood Mask\n{len(gdf)} features, {flood_mask.sum():,} pixels", 
                    fontsize=11, fontweight='bold', color='#2c3e50'
                )
                axes[idx, 2].axis('off')
                
                # Plot 4: Combined overlay
                axes[idx, 3].imshow(rgb_normalized)
                
                # Overlay permanent water in blue
                if water_mask is not None:
                    water_rgba = np.zeros((*water_mask.shape, 4))
                    water_rgba[water_mask == 1] = [0, 0, 1, 0.4]  # Blue with 40% transparency
                    axes[idx, 3].imshow(water_rgba)
                
                # Overlay flood mask in red
                flood_rgba = np.zeros((*flood_mask.shape, 4))
                flood_rgba[flood_mask == 1] = [1, 0, 0, 0.5]  # Red with 50% transparency
                axes[idx, 3].imshow(flood_rgba)
                
                # Add legend
                legend_elements = [
                    Patch(facecolor='red', alpha=0.5, label='Flooded Area'),
                    Patch(facecolor='blue', alpha=0.4, label='Permanent Water'),
                    Patch(facecolor='none', label=f'{len(gdf)} flood polygons')
                ]
                axes[idx, 3].legend(handles=legend_elements, loc='upper right', 
                                   fontsize=9, framealpha=0.9)
                
                axes[idx, 3].set_title(
                    f"RGB + Water + Flood Overlay", 
                    fontsize=11, fontweight='bold', color='#2c3e50'
                )
                axes[idx, 3].axis('off')
            else:
                axes[idx, 2].text(0.5, 0.5, f"GeoJSON not found:\n{geojson_path}", 
                                 ha='center', va='center', fontsize=10, color='red')
                axes[idx, 2].axis('off')
                axes[idx, 3].axis('off')
        
        plt.tight_layout(pad=2.0)
        os.makedirs('./exp/plots', exist_ok=True)
        plt.savefig('./exp/plots/timor_leste_with_permanent_water.png', dpi=150, bbox_inches='tight')
        print("Saved: ./exp/plots/timor_leste_with_permanent_water.png")
        plt.close()


def plot_single_combined_with_water(tif_path, geojson_path, watermask_path, satellite_type, bands_to_display=(0, 1, 2)):
    """Plot a single TIF file with its GeoJSON mask and permanent water."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # Load satellite data
    sat_data = load_satellite_data(tif_path, satellite_type, bands_to_display)
    rgb_normalized = sat_data['rgb_normalized']
    
    # Plot RGB
    axes[0].imshow(rgb_normalized)
    axes[0].set_title(f"RGB Image ({satellite_type})", fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    # Load and plot permanent water
    water_mask = load_permanent_water_mask(watermask_path)
    if water_mask is not None:
        axes[1].imshow(water_mask, cmap='Blues', interpolation='nearest')
        axes[1].set_title(f"Permanent Water ({water_mask.sum():,} pixels)", fontsize=13, fontweight='bold')
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, "Water mask not found", ha='center', va='center', fontsize=10, color='red')
        axes[1].axis('off')
    
    # Load GeoJSON mask
    flood_mask, gdf = load_geojson_mask(
        geojson_path, 
        sat_data['height'], 
        sat_data['width'], 
        sat_data['transform'], 
        sat_data['crs']
    )
    
    if flood_mask is not None:
        # Plot flood mask
        axes[2].imshow(flood_mask, cmap='Reds', interpolation='nearest')
        axes[2].set_title(f"Flood Mask ({len(gdf)} features)", fontsize=13, fontweight='bold')
        axes[2].axis('off')
        
        # Plot overlay
        axes[3].imshow(rgb_normalized)
        
        if water_mask is not None:
            water_rgba = np.zeros((*water_mask.shape, 4))
            water_rgba[water_mask == 1] = [0, 0, 1, 0.4]
            axes[3].imshow(water_rgba)
        
        flood_rgba = np.zeros((*flood_mask.shape, 4))
        flood_rgba[flood_mask == 1] = [1, 0, 0, 0.5]
        axes[3].imshow(flood_rgba)
        
        legend_elements = [
            Patch(facecolor='red', alpha=0.5, label='Flooded Area'),
            Patch(facecolor='blue', alpha=0.4, label='Permanent Water'),
            Patch(facecolor='none', label=f'{len(gdf)} polygons\n{flood_mask.sum():,} pixels')
        ]
        axes[3].legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
        axes[3].set_title("Combined Overlay", fontsize=13, fontweight='bold')
        axes[3].axis('off')
    else:
        axes[2].text(0.5, 0.5, f"GeoJSON not found:\n{geojson_path}", 
                    ha='center', va='center', fontsize=10, color='red')
        axes[2].axis('off')
        axes[3].axis('off')
    
    plt.tight_layout()
    os.makedirs('./exp/plots', exist_ok=True)
    safe_name = os.path.basename(tif_path).replace('.tif', '')
    plt.savefig(f'./exp/plots/{safe_name}_with_water.png', dpi=150, bbox_inches='tight')
    print(f"Saved: ./exp/plots/{safe_name}_with_water.png")
    plt.close()


# ========== MAIN EXECUTION ==========

# Create combined triplets
combined_triplets = [
    ((tif_path, geojson_path, watermask_path), satellite_type) 
    for (tif_path, sat1), (geojson_path, sat2), (watermask_path, sat3), satellite_type in 
    zip(files, files_geojson, files_watermask, timor_leste_events.values())
]

# Plot all in one figure
plot_tif_with_geojson_and_water(combined_triplets)