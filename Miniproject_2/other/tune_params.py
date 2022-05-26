# -*- coding: utf-8 -*-
"""
Created on Wed May 18 07:39:47 2022

@author: manon
"""
import os
import sys
import torch
from ImgDataset import ImgDataset
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIR)

from path import create_dirs
create_dirs()
from path import DATA_TRAIN, DATA_VALID

from Model import Model

#%% Load Training and Testing set
# Tuning will be done using 1000 images on 10 epochs

data_train = ImgDataset(DATA_TRAIN,nbsamples=1000)
xtrain, ytrain = data_train[:]

data_valid = ImgDataset(DATA_VALID,nbsamples=100)
xvalid, yvalid = data_valid[:]

#%% Tune number of channels

channels = [32,64,128,256]
avg_losses_channels = np.zeros((4,4))
avg_pnrs_channels = np.zeros((4,4))
nb_params = np.zeros((4,4))

for i,channel1 in enumerate(channels):
    for j,channel2 in enumerate(channels):
        model = Model(input_channels=3, output_channels=3,kernel_size = 3,
                      nbchannels1=channel1,nbchannels2=channel2,nbchannels3=channel1)
        model.train(xtrain, ytrain, 10, save=False)
        avg_losses_channels[i,j], avg_pnrs_channels[i,j] = model.validation(xvalid, yvalid)
        nb_params[i,j]=model.nb_params
    
#%% Scatter plot
from adjustText import adjust_text

fig,ax = plt.subplots(figsize=(5, 4))
ax.scatter(nb_params,array_pnst_channels,c='tomato')
ax.set_xlim(left=-0.2e6, right=None)
ax.set_ylim(bottom=21.4, top=None)
texts=[]
for i,channel1 in enumerate(channels):
    for j,channel2 in enumerate(channels):
        txt = [channel1,channel2]
        #ax.annotate(txt, (nb_params[i,j], array_pnst_channels[i,j]))     
        texts.append(ax.annotate(txt, xy=(nb_params[i,j], array_pnst_channels[i,j]), xytext=(nb_params[i,j], array_pnst_channels[i,j])))
adjust_text(texts)
ax.set_xlabel("Number of parameters")
ax.set_ylabel("PNSR [dB]")
ax.grid() 
plt.savefig('scatter_PNSR_nb_params.eps', bbox_inches='tight', format='eps')
#plt.tight_layout()  
