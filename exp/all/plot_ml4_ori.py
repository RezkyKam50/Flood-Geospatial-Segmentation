import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

ROOT = 'datasets/WorldFloodsv2'
TEST_PATH_S2 = f'{ROOT}/train/S2/'
EXTENSION = '.tif'

TIMOR_LESTE_EVENTS = {
    "EMSR507_AOI01_DEL_PRODUCT": "Pleiades-1A-1B",
    "EMSR507_AOI02_DEL_PRODUCT": "PlanetScope",
    "EMSR507_AOI03_DEL_PRODUCT": "PlanetScope",
    "EMSR507_AOI05_DEL_PRODUCT": "Sentinel-2",
    "EMSR507_AOI07_GRA_PRODUCT": "PlanetScope"
}

# 'names': ['Coastal Blue', 'Blue', 'Green I', 'Green', 
#          'Yellow', 'Red', 'RedEdge', 'NIR']

# RGB band indices for each satellite (Red, Green, Blue order)
SATELLITE_RGB_BANDS = {
    'Sentinel-2': [4, 3, 2],      # Red, Green, Blue
    'Pleiades-1A-1B': [4, 3, 2],  # Red, Green, Blue
    'PlanetScope': [5, 4, 2]      # Red, Green I, Blue
}

def load_rgb_bands(img_path, satellite):
    """Load RGB bands specific to the satellite type"""
    rgb_bands = SATELLITE_RGB_BANDS[satellite]
    
    with rasterio.open(img_path) as src:
        img = src.read(rgb_bands)
    
    rgb = np.transpose(img, (1, 2, 0)).astype(np.float32)
    
    # Normalize to [0, 1]
    rgb_min = rgb.min()
    rgb_max = rgb.max()
    if rgb_max > rgb_min:
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
    
    return rgb

def plot_timor_leste_samples():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (event_id, satellite) in enumerate(TIMOR_LESTE_EVENTS.items()):
        img_path = f"{TEST_PATH_S2}{event_id}{EXTENSION}"
        rgb = load_rgb_bands(img_path, satellite)
        
        axes[idx].imshow(rgb)
        axes[idx].set_title(f"{event_id}\n{satellite}", fontsize=9)
        axes[idx].axis('off')
    
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig('timor_leste_original.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plot_timor_leste_samples()