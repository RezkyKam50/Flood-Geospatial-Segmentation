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
PATCH_SIZE = 224  # New patch size for cutting images
MEANS = [0.13692222, 0.13376727, 0.11943894, 0.30450596, 0.20170933, 0.11685023]
STDS = [0.03381057, 0.03535441, 0.04496607, 0.07556641, 0.06130259, 0.04689224]

# this is for testing, using ml4floods
root = 'datasets/WorldFloodsv2'
metadata_path = f'{root}/dataset_metadata.csv'
test_path_label = f'{root}/train/PERMANENTWATERJRC/'    # comparable to 'LabelHand' in sen1floods11
test_path_s2 = f'{root}/train/S2/'                      # comparable to 'S2L1CHand' in sen1floods11
# Unique class values: [0 1 2 3]
# Number of classes: 4
# 0: Invalid/No Data
# 1: Land (non-flooded)
# 2: Water (flood water)
# 3: Permanent Water (from JRC permanent water layer)


# this is for training, using sen1floods11. It contains 'bolivia' for testing but we'll use 'timor-leste' from ml4floods for testing
LABEL_DIR = 'data/LabelHand'
IMAGE_DIR = 'data/S2L1CHand'
DATASET_DIR = 'splits'
# Unique class values: [-1  0  1]
# Number of classes: 3
# -1: No Data / Invalid / Clouds
# 0: Land (Not Water)
# 1: Water (includes both flood water and permanent water)


# this is the .tif for testing, using ml4floods 'timor-leste' data
extension = '.tif'
timor_leste_events = {
    "EMSR507_AOI01_DEL_PRODUCT": "Pleiades-1A-1B",
    "EMSR507_AOI02_DEL_PRODUCT": "PlanetScope",
    "EMSR507_AOI03_DEL_PRODUCT": "PlanetScope",
    "EMSR507_AOI05_DEL_PRODUCT": "Sentinel-2",
    "EMSR507_AOI07_GRA_PRODUCT": "PlanetScope"
}

test_files_label = [(f"{test_path_label}{event_id}{extension}", satellite) 
         for event_id, satellite in timor_leste_events.items()]

test_files_s2 = [(f"{test_path_s2}{event_id}{extension}", satellite) 
         for event_id, satellite in timor_leste_events.items()]

USED_BANDS = (1,2,3,8,11,12)

def processAndAugment(data):
    img,label = data
    img = img[USED_BANDS, :, :].astype(np.float32)
    label = label.squeeze().astype(np.int16)
    
    img, label = torch.tensor(img), torch.tensor(label)
    norm = transforms.Normalize(MEANS, STDS)
    img = norm(img)

    # Get params for random transforms
    i, j, h, w = transforms.RandomCrop.get_params(img, (INPUT_SIZE, INPUT_SIZE))
    
    img = F.crop(img, i, j, h, w)
    label = F.crop(label, i, j, h, w)
    if random.random() > 0.5:
        img = F.hflip(img)
        label = F.hflip(label)
    if random.random() > 0.5:
        img = F.vflip(img)
        label = F.vflip(label)

    return img, label


def processTestData(data):
    img,label = data
    img = img[USED_BANDS, :, :].astype(np.float32)
    label = label.squeeze().astype(np.int16)
    
    img, label = torch.tensor(img), torch.tensor(label)
    norm = transforms.Normalize(MEANS, STDS)
    img = norm(img)
    
    ims = [F.crop(img, 0, 0, INPUT_SIZE, INPUT_SIZE), F.crop(img, 0, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE),
                F.crop(img, INPUT_SIZE, 0, INPUT_SIZE, INPUT_SIZE), F.crop(img, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)]
    labels = [F.crop(label, 0, 0, INPUT_SIZE, INPUT_SIZE), F.crop(label, 0, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE),
                F.crop(label, INPUT_SIZE, 0, INPUT_SIZE, INPUT_SIZE), F.crop(label, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)]
    
    ims = torch.stack(ims)
    labels = torch.stack([label.squeeze() for label in labels])
    
    return ims, labels

def processTimorLesteData(data):
    """
    Process Timor-Leste data by cutting into 224x224 patches.
    Handles varying resolutions by creating patches from the entire image.
    """
    img, label = data
    img = img[USED_BANDS, :, :].astype(np.float32)
    label = label.squeeze().astype(np.int16)
    
    img, label = torch.tensor(img), torch.tensor(label)
    norm = transforms.Normalize(MEANS, STDS)
    img = norm(img)
    
    # Get image dimensions
    _, h, w = img.shape
    
    # Calculate number of patches in each dimension
    n_patches_h = h // PATCH_SIZE
    n_patches_w = w // PATCH_SIZE
    
    # Extract all non-overlapping patches
    ims = []
    labels = []
    
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            top = i * PATCH_SIZE
            left = j * PATCH_SIZE
            
            img_patch = F.crop(img, top, left, PATCH_SIZE, PATCH_SIZE)
            label_patch = F.crop(label, top, left, PATCH_SIZE, PATCH_SIZE)
            
            ims.append(img_patch)
            labels.append(label_patch)
    
    # Stack all patches
    if len(ims) > 0:
        ims = torch.stack(ims)
        labels = torch.stack(labels)
    else:
        # If image is smaller than PATCH_SIZE, return empty tensors
        ims = torch.empty(0, img.shape[0], PATCH_SIZE, PATCH_SIZE)
        labels = torch.empty(0, PATCH_SIZE, PATCH_SIZE, dtype=torch.int16)
    
    return ims, labels
  
def processTestIm(img, bands):
    img = img[bands, :, :].astype(np.float32)
    img = torch.tensor(img)
    norm = transforms.Normalize(MEANS, STDS)
    img = norm(img)
    return img.unsqueeze(0)

def getArrFlood(fname):
  return rasterio.open(fname).read()

def load_flood_data(path, dataset_type):
    fpath = os.path.join(path, DATASET_DIR, f'flood_{dataset_type}_data.txt')
    with open(fpath) as f:
        get_img_path = lambda identifier: os.path.join(path, IMAGE_DIR, f"{identifier}_S2Hand.tif")
        get_label_path = lambda identifier: os.path.join(path, LABEL_DIR, f"{identifier}_LabelHand.tif")
        
        # Read single column of identifiers
        data_files = []
        for line in f:
            identifier = line.strip()
            if identifier:  # Skip empty lines
                data_files.append((get_img_path(identifier), get_label_path(identifier)))
    
    return download_flood_water_data_from_list(data_files)

def load_timor_leste_data():
    """Load Timor-Leste test data from ml4floods dataset using explicitly listed files"""
    data_files = []
    for event_id, satellite in timor_leste_events.items():
        img_path = f"{test_path_s2}{event_id}{extension}"
        label_path = f"{test_path_label}{event_id}{extension}"
        data_files.append((img_path, label_path))
    
    return download_flood_water_data_from_list(data_files)
 
def download_flood_water_data_from_list(l):
  flood_data = []
  for (im_path, mask_path) in l:
    if not os.path.exists(im_path) or not os.path.exists(mask_path):
      raise ValueError(f"File not found: {im_path} or {mask_path}")
    arr_x = np.nan_to_num(getArrFlood(im_path))
    arr_y = getArrFlood(mask_path)
    
    # Handle different label schemes
    # sen1floods11: -1 (invalid), 0 (land), 1 (water)
    # ml4floods: 0 (invalid), 1 (land), 2 (water), 3 (permanent water)
    
    # Check if this is ml4floods data (has values 2 or 3)
    if np.any((arr_y == 2) | (arr_y == 3)):
      # ml4floods label conversion:
      # 0 (invalid) -> 255 (ignore)
      # 1 (land) -> 0 (land/non-water)
      # 2 (water) -> 1 (water)
      # 3 (permanent water) -> 1 (water)
      arr_y_new = np.zeros_like(arr_y)
      arr_y_new[arr_y == 0] = 255  # invalid -> ignore
      arr_y_new[arr_y == 1] = 0    # land -> land
      arr_y_new[arr_y == 2] = 1    # water -> water
      arr_y_new[arr_y == 3] = 1    # permanent water -> water
      arr_y = arr_y_new
    else:
      # sen1floods11 label conversion:
      # -1 (invalid) -> 255 (ignore)
      arr_y[arr_y == -1] = 255

    flood_data.append((arr_x, arr_y))
  return flood_data

def get_train_loader(data_path, args):
    train_data = load_flood_data(data_path, 'train')
    train_dataset = InMemoryDataset(train_data, processAndAugment)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, sampler=None,
                    batch_sampler=None, num_workers=0, collate_fn=None,
                    pin_memory=True, drop_last=False, timeout=0,
                    worker_init_fn=None)
    return train_loader
    
def get_test_loader(data_path, type):
    valid_data = load_flood_data(data_path, type)
    valid_dataset = InMemoryDataset(valid_data, processTestData)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=3, shuffle=True, sampler=None,
                    batch_sampler=None, num_workers=0, collate_fn=lambda x: (torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0)),
                    pin_memory=True, drop_last=False, timeout=0,
                    worker_init_fn=None)
    return valid_loader

def get_timor_leste_loader():
    """Get data loader for Timor-Leste test set using 224x224 patches"""
    timor_leste_data = load_timor_leste_data()
    timor_leste_dataset = InMemoryDataset(timor_leste_data, processTimorLesteData)
    timor_leste_loader = torch.utils.data.DataLoader(timor_leste_dataset, batch_size=3, shuffle=False, sampler=None,
                    batch_sampler=None, num_workers=0, collate_fn=lambda x: (torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0)),
                    pin_memory=True, drop_last=False, timeout=0,
                    worker_init_fn=None)
    return timor_leste_loader

def get_loader(data_path, type, args):
    if type == 'timor_leste':
        return get_timor_leste_loader()
    elif type == 'train':
        return get_train_loader(data_path, args)
    else:
        return get_test_loader(data_path, type)