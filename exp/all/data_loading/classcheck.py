import rasterio
import numpy as np

# Assuming labels are in a similar path structure
root_ml4 = 'datasets/WorldFloodsv2'
label_path = f'{root_ml4}/train/PERMANENTWATERJRC/EMSR507_AOI01_DEL_PRODUCT.tif'

with rasterio.open(label_path) as src:
    label_data = src.read(1)  # Read first band
    unique_classes = np.unique(label_data)
    num_classes = len(unique_classes)
    
    print(f"Unique class values: {unique_classes}")
    print(f"Number of classes: {num_classes}")
    print(f"Class distribution: {np.bincount(label_data.flatten().astype(int))}")


root_sens1 = 'datasets/sen1floods11_v1.1'
label_path = f'{root_sens1}/data/LabelHand/Bolivia_23014_LabelHand.tif'

with rasterio.open(label_path) as src:
    label_data = src.read(1)  # Read first band
    unique_classes = np.unique(label_data)
    num_classes = len(unique_classes)
    
    print(f"Unique class values: {unique_classes}")
    print(f"Number of classes: {num_classes}")
    # print(f"Class distribution: {np.bincount(label_data.flatten().astype(int))}")