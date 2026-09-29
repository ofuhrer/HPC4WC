import numpy as np
import pandas as pd
import xarray as xr
import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch

class T_2M_Dataset(Dataset):

    def __init__(self, root_dir, in_len, out_len, field="T_2M", transform=None):
        """
        Args:
            root_dir (str): Directory with all images, named/sorted so that each
                            consecutive 10 images form one series.
            transform (callable, optional): Optional transform to be applied
                                            on an image.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Open dataset
        self.data_store = xr.open_zarr(root_dir, consolidated=True)
        self.ds = self.data_store[field]

        self.ts = self.data_store.time.values
        self.time_len = self.ts.shape[0]
        self.in_len = in_len
        self.out_len = out_len
        self.seq_len = in_len+out_len
        self.time_blocks = self.gen_time_block()

        self.begin_time = self.ts[0]
        self.end_time = self.ts[-1]
        self.current_time = self.begin_time

    def __len__(self):
        return len(self.time_blocks)

    def __getitem__(self, idx):

        time_clip = self.time_blocks[idx]
        temp_fields = self.ds.sel(time=time_clip).values
        
        temp_fields = torch.from_numpy(temp_fields)
        if self.transform:
            temp_fields = self.transform(temp_fields)

        temp_fields = temp_fields#.unsqueeze(1)
        
        inputs = temp_fields[:self.in_len,:,:]    # shape: (T, C, H, W)
        targets = temp_fields[self.in_len:self.in_len+self.out_len,:,:]  # shape: (T, C, H, W)

        return inputs, targets
    
    def gen_time_block(self):
        time_blocks = [
            self.ts[i:i + self.seq_len]
            for i in range(0, self.time_len-self.seq_len+1)
        ]
        return time_blocks
