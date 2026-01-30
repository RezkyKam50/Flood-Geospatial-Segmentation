import torch
import rasterio
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, preprocess_func):
        self.data_list = data_list
        self.preprocess_func = preprocess_func

    def __getitem__(self, i):
        return self.preprocess_func(self.data_list[i])

    def __len__(self):
        return len(self.data_list)


# Satellite band mappings
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
PATCH_SIZE = 224
STRIDE = 224

root = 'datasets/WorldFloodsv2'
test_path_s2 = f'{root}/train/S2/'
test_path_labels = f'{root}/train/gt/'  
 

extension = '.tif'

timor_leste_events = {
    "EMSR507_AOI01_DEL_PRODUCT": "Pleiades-1A-1B",
    "EMSR507_AOI02_DEL_PRODUCT": "PlanetScope",
    "EMSR507_AOI03_DEL_PRODUCT": "PlanetScope",
    "EMSR507_AOI05_DEL_PRODUCT": "Sentinel-2",
    "EMSR507_AOI07_GRA_PRODUCT": "PlanetScope"
}

files_s2 = [(f"{test_path_s2}{event_id}{extension}", satellite) 
            for event_id, satellite in timor_leste_events.items()]

files_gt = [(f"{test_path_labels}{event_id}{extension}", satellite) 
            for event_id, satellite in timor_leste_events.items()]

 
output_root_s2 = "./datasets/Timor_Processed/S2"
os.makedirs(output_root_s2, exist_ok=True)

output_root_gt = "./datasets/Timor_Processed/GT"
os.makedirs(output_root_gt, exist_ok=True)

output_root_floodmask = "./datasets/Timor_Processed/Floodmask"
os.makedirs(output_root_gt, exist_ok=True)

def sliding_window_crop(image, window_size=PATCH_SIZE, stride=STRIDE):
    C, H, W = image.shape
    patches = []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + window_size, H)
            x_end = min(x + window_size, W)
            y_start = max(y_end - window_size, 0)
            x_start = max(x_end - window_size, 0)
            patch = image[:, y_start:y_end, x_start:x_end]
            patches.append(patch)
    return patches


def read_tif_as_tensor(tif_path):
    with rasterio.open(tif_path) as src:
        img = src.read()  # shape: (bands, H, W)
        img = torch.from_numpy(img).float()
    return img


def save_patch_as_tif(patch_tensor, output_path):
    patch_np = patch_tensor.numpy()
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=patch_np.shape[1],
        width=patch_np.shape[2],
        count=patch_np.shape[0],
        dtype=patch_np.dtype
    ) as dst:
        dst.write(patch_np)


def plot_patches(patches, cols=5, save_path=None, is_label=False):
    rows = (len(patches) + cols - 1) // cols
    patch_images = []
    font = ImageFont.load_default()

    for idx, patch in enumerate(patches):
        if is_label:
            # Labels assumed to be single-channel
            patch_np = patch[0].numpy()
            patch_np = ((patch_np - patch_np.min()) / (patch_np.max() - patch_np.min() + 1e-8) * 255).astype(np.uint8)
            img = Image.fromarray(patch_np).convert("L")
        else:
            # RGB visualization for images
            patch_np = patch[:3].numpy()
            patch_np = (patch_np - patch_np.min()) / (patch_np.max() - patch_np.min() + 1e-8) * 255
            patch_np = patch_np.transpose(1,2,0).astype(np.uint8)
            img = Image.fromarray(patch_np)
        draw = ImageDraw.Draw(img)
        draw.text((5,5), str(idx), fill=(255,0,0), font=font)
        patch_images.append(img)

    width, height = patch_images[0].size
    grid_img = Image.new('RGB' if not is_label else 'L', (cols * width, rows * height), color=(255,255,255) if not is_label else 255)
    for i, img in enumerate(patch_images):
        row = i // cols
        col = i % cols
        grid_img.paste(img, (col*width, row*height))

    if save_path:
        grid_img.save(save_path)

# Class color map: 0=invalid, 1=land, 2=flood, 3=permanent water
CLASS_COLORS = {
    0: (0, 0, 0),       # black for invalid/no data
    1: (34, 139, 34),   # green for flood (gt)
    2: (0, 0, 255),     # blue for cloud (gt)
}

def plot_label_patches(label_patches, cols=5, save_path=None):
    rows = (len(label_patches) + cols - 1) // cols
    patch_images = []
    font = ImageFont.load_default()

    for idx, patch in enumerate(label_patches):
        patch_np = patch[0].numpy().astype(int)  # assume single channel
        H, W = patch_np.shape
        color_img = np.zeros((H, W, 3), dtype=np.uint8)
        for cls, color in CLASS_COLORS.items():
            color_img[patch_np == cls] = color
        img = Image.fromarray(color_img)
        draw = ImageDraw.Draw(img)
        draw.text((5,5), str(idx), fill=(255,0,0), font=font)
        patch_images.append(img)

    width, height = patch_images[0].size
    grid_img = Image.new('RGB', (cols * width, rows * height), color=(255,255,255))
    for i, img in enumerate(patch_images):
        row = i // cols
        col = i % cols
        grid_img.paste(img, (col*width, row*height))

    if save_path:
        grid_img.save(save_path)


# Main processing loop
for tif_path, satellite in files_s2:
    print(f"Processing {tif_path} ({satellite})...")
    img_tensor = read_tif_as_tensor(tif_path)
    patches = sliding_window_crop(img_tensor, PATCH_SIZE, STRIDE)

    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    patch_output_dir = os.path.join(output_root_s2, base_name)
    os.makedirs(patch_output_dir, exist_ok=True)

    # Save image patches
    for idx, patch in enumerate(patches):
        patch_name = f"{base_name}_{idx}.tif"
        save_patch_as_tif(patch, os.path.join(patch_output_dir, patch_name))

    # # Plot image patches
    # plot_save_path = os.path.join(patch_output_dir, f"{base_name}_grid.png")
    # plot_patches(patches, save_path=plot_save_path)

    # # If labels exist in a corresponding folder
    # label_path = tif_path.replace('/S2/', '/gt/')  # assuming label folder structure
    # if os.path.exists(label_path):
    #     label_tensor = read_tif_as_tensor(label_path)
    #     label_patches = sliding_window_crop(label_tensor, PATCH_SIZE, STRIDE)
    #     plot_label_path = os.path.join(patch_output_dir, f"{base_name}_labels_grid.png")
    #     plot_label_patches(label_patches, save_path=plot_label_path)


# Main processing loop
for tif_path, satellite in files_gt:
    print(f"Processing {tif_path} ({satellite})...")
    img_tensor = read_tif_as_tensor(tif_path)
    patches = sliding_window_crop(img_tensor, PATCH_SIZE, STRIDE)

    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    patch_output_dir = os.path.join(output_root_gt, base_name)
    os.makedirs(patch_output_dir, exist_ok=True)

    # Save image patches
    for idx, patch in enumerate(patches):
        patch_name = f"{base_name}_{idx}.tif"
        save_patch_as_tif(patch, os.path.join(patch_output_dir, patch_name))

    # # Plot image patches
    # plot_save_path = os.path.join(patch_output_dir, f"{base_name}_grid.png")
    # plot_patches(patches, save_path=plot_save_path)

    # # If labels exist in a corresponding folder
    # label_path = tif_path.replace('/S2/', '/gt/')  # assuming label folder structure
    # if os.path.exists(label_path):
    #     label_tensor = read_tif_as_tensor(label_path)
    #     label_patches = sliding_window_crop(label_tensor, PATCH_SIZE, STRIDE)
    #     plot_label_path = os.path.join(patch_output_dir, f"{base_name}_labels_grid.png")
    #     plot_label_patches(label_patches, save_path=plot_label_path)

