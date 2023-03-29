import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import time
import random
from PIL import Image
import openpyxl
import random

import torch
import torch.nn as nn  
from torch.utils.data import Dataset, DataLoader
from data import load_npz, return_class_list, _generate_hm_new

from scipy.ndimage import zoom

class BoundingBoxDataSet(Dataset):
    def __init__(self, dir_data, flag_shuffle, sigma=4):
        '''
        dir_data : directory containing npz files
        '''
        self.dir_data = dir_data
        
        self.sigma = sigma
        
        # if flag_balance == False:
            # self.slice_list = os.listdir(dir_data)
        # slice_list_zero, slice_list_nonzero = return_class_list(dir_data)
        # slice_list_zero_selected = random.sample(slice_list_zero, int(len(slice_list_nonzero) * flag_balance))
        # self.slice_list = []
        # self.slice_list.extend(slice_list_nonzero)
        # self.slice_list.extend(slice_list_zero_selected)
        # self.slice_list.sort()

        self.slice_list = return_class_list(dir_data)
         
        if flag_shuffle: 
            np.random.seed(42)
            np.random.shuffle(self.slice_list)
        
        

    def __len__(self):
        return len(self.slice_list)
    
    def __getitem__(self, index):
        slice_number = self.slice_list[index]
        path_slice = os.path.join(self.dir_data, slice_number)

        slice_number = slice_number[:slice_number.rfind('.npz')]
     
        ct, min_x, min_y, max_x, max_y, _, spacing_original, thickness_original = load_npz(path_slice)
        ## size 512 -> 128
        # ct = zoom(ct, 0.25)
        # min_x /= 4.0
        # min_y /= 4.0
        # max_x /= 4.0
        # max_y /= 4.0

        hms = _generate_hm_new(128, 128, np.array([[min_x,min_y],[max_x, max_y]]), sigma=self.sigma)
        ct_tensor = torch.from_numpy(np.array([ct])).type(torch.FloatTensor)
        hms_tensor = torch.from_numpy(hms).type(torch.FloatTensor)

        sample = {'ct': ct_tensor, 'hms' : hms_tensor, 'landmarks':np.array([[min_x,min_y],[max_x, max_y]]), 'number':slice_number}

        return sample


if __name__ == '__main__':
    dir_data = "Z:/Backup/Users/kys/BoundingBox/data/processed/heatmap_10_128_fixed/train/"

    train_dataset = BoundingBoxDataSet(dir_data, flag_shuffle=False, flag_balance=True)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0)
    for batch in train_loader:
        print(len(train_loader.dataset))
        input_tensor = batch['ct']
        gt_tensor = batch['hms']
        landmarks = batch['landmarks']

        print(input_tensor.shape)
        print(gt_tensor.shape)
        print(landmarks)

        exit()

