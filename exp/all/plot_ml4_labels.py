import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import ListedColormap

root = 'datasets/WorldFloodsv2'
test_path_label = f'{root}/train/PERMANENTWATERJRC/'
test_path_s2 = f'{root}/train/S2/'
extension = '.tif'

SATELLITE_ALL_BANDS_MAPPING = {
    'Sentinel-2': {
        'bands': [1, 2, 3, 8, 11, 12],  
        'names': ['Coastal Aerosol', 'Blue', 'Green', 'Red', 
                 'Red Edge 1', 'Red Edge 2', 'Red Edge 3', 'NIR', 
                 'Narrow NIR', 'Water Vapour', 'Cirrus', 'SWIR1', 'SWIR2']
    },
    'Pleiades-1A-1B': {
        'bands': [4, 3, 2],   
        'names': ['Panchromatic', 'Blue', 'Green', 'Red', 'NIR']
    },
    'PlanetScope': {
        'bands': [5, 4, 2], 
        'names': ['Coastal Blue', 'Blue', 'Green I', 'Green', 
                 'Yellow', 'Red', 'RedEdge', 'NIR']
    }
}

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

timor_leste_events = {
    "EMSR507_AOI01_DEL_PRODUCT": "Pleiades-1A-1B",
    "EMSR507_AOI02_DEL_PRODUCT": "PlanetScope",
    "EMSR507_AOI03_DEL_PRODUCT": "PlanetScope",
    "EMSR507_AOI05_DEL_PRODUCT": "Sentinel-2",
    "EMSR507_AOI07_GRA_PRODUCT": "PlanetScope"
}

for event_id, satellite in timor_leste_events.items():
    img_path = f"{test_path_s2}{event_id}{extension}"
    label_path = f"{test_path_label}{event_id}{extension}"
    
    with rasterio.open(img_path) as src:
        img = src.read()

    rgb_img = load_rgb_bands(img_path, satellite)
    
    with rasterio.open(label_path) as src:
        label = src.read().squeeze()
    
    
    cloud_mask = (label == 0).astype(np.uint8)
    land_mask = (label == 1).astype(np.uint8)
    flood_mask = (label == 2).astype(np.uint8)
    permanent_water_mask = (label == 3).astype(np.uint8)
    
    combined = np.zeros_like(label)
    combined[label == 0] = 0
    combined[label == 1] = 1
    combined[label == 2] = 2
    combined[label == 3] = 3
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    axes[0].imshow(rgb_img)
    axes[0].set_title(f'Original Image - {satellite}')
    axes[0].axis('off')
    
    cmap_flood = ListedColormap(['black', 'red'])
    axes[1].imshow(flood_mask, cmap=cmap_flood, vmin=0, vmax=1)
    axes[1].set_title('Flood Label')
    axes[1].axis('off')
    
    cmap_cloud = ListedColormap(['black', 'gray'])
    axes[2].imshow(cloud_mask, cmap=cmap_cloud, vmin=0, vmax=1)
    axes[2].set_title('Cloud/Invalid Label')
    axes[2].axis('off')
    
    cmap_perm = ListedColormap(['black', 'blue'])
    axes[3].imshow(permanent_water_mask, cmap=cmap_perm, vmin=0, vmax=1)
    axes[3].set_title('Permanent Water Label')
    axes[3].axis('off')
    
    cmap_combined = ListedColormap(['gray', 'green', 'red', 'blue'])
    im = axes[4].imshow(combined, cmap=cmap_combined, vmin=0, vmax=3)
    axes[4].set_title('Combined')
    axes[4].axis('off')
    
    cbar = plt.colorbar(im, ax=axes[4], ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(['Invalid', 'Non-flood', 'Flood', 'Permanent Water'])
    
    plt.suptitle(f'{event_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'timor_leste_{event_id}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Saved: timor_leste_{event_id}.png')