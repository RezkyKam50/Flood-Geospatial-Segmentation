import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import random
from time import time
import csv
import os
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

USED_BANDS = (1,2,3,8,11,12)
INPUT_SIZE = 224

MEANS = [0.14245495, 0.13921481, 0.12434631, 0.31420089, 0.20743526,0.12046503]
STDS = [0.04036231, 0.04186983, 0.05267646, 0.0822221 , 0.06834774, 0.05294205]

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
  
def processTestIm(img, bands):
    img = img[bands, :, :].astype(np.float32)
    img = torch.tensor(img)
    norm = transforms.Normalize(MEANS, STDS)
    img = norm(img)
    return img.unsqueeze(0)

def getArrFlood(fname):
  return rasterio.open(fname).read()

LABEL_DIR = 'data/LabelHand'
IMAGE_DIR = 'data/S2L1CHand'
DATASET_DIR = 'splits'

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
 
def download_flood_water_data_from_list(l):
  flood_data = []
  for (im_path, mask_path) in l:
    if not os.path.exists(im_path) or not os.path.exists(mask_path):
      raise ValueError(f"File not found: {im_path} or {mask_path}")
    arr_x = np.nan_to_num(getArrFlood(im_path))
    arr_y = getArrFlood(mask_path)
    arr_y[arr_y == -1] = 255 

    flood_data.append((arr_x,arr_y))
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

def get_loader(data_path, type, args):
    if type == 'train':
        return get_train_loader(data_path, args)
    else:
        return get_test_loader(data_path, type)
  