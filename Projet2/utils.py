# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:17:05 2022

@author: manon
"""
import torchvision
import matplotlib.pyplot as plt
import numpy as np



def show(imgs):
    '''Plot the images
    imgs: list of images'''

    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()
