# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:23:38 2022

@author: manon
"""

import torch
from torch.utils.data import Dataset

class ImgDataset(Dataset):
    """Dataset of nuclear fusion time series."""
    def __init__(self, data_dir: str,nbsamples=1000) -> None:
        self.data_dir = data_dir
        self.noisy_imgs1 ,self.noisy_imgs2 = torch.load(data_dir)
        self.noisy_imgs1 = self.noisy_imgs1[0:nbsamples, ...]
        self.noisy_imgs2 = self.noisy_imgs2[0:nbsamples, ...]

    def __getitem__(self, index):
        x = self.noisy_imgs1[index, ...]/255.0
        y = self.noisy_imgs2[index, ...]/255.0
        return x, y

    def __len__(self):
        return self.noisy_imgs1.shape[0]