# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:23:38 2022

@author: manon
"""

#%load_ext autoreload
#%autoreload 2

import os
import sys
import torch
from ImgDataset import ImgDataset
from utils import show

ROOT_DIR = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIR)

from path import create_dirs
create_dirs()
from path import DATA_TRAIN, DATA_VALID
#%%
from Model import Model
model = Model(input_channels=3, output_channels=3,kernel_size = 7,
                      nbchannels1=256,nbchannels2=32,nbchannels3=256)
        
#%% run this cell to load pretrained model
model.load_pretrained_model()

#%% run this cell to train model
data_train = ImgDataset(DATA_TRAIN,nbsamples=1000)
x, y = data_train[:]

#%%
model.train(x, y, 10, save=True)

#%% run this cell to valid model
data_valid = ImgDataset(DATA_VALID,nbsamples=100)
x, y = data_valid[:]
model.validation(x, y)

#%% save model
model.save_model(name='modelkernel7.pickle')