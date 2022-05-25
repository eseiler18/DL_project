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

    fix, ax = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, im in enumerate(imgs):
        im = im.detach()
        im = torchvision.transforms.functional.to_pil_image(im)
        ax[0, i].imshow(np.asarray(im))
        ax[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    ax[0,0].set_title('Noisy')
    ax[0,1].set_title('Denoised')
    ax[0,2].set_title('Clean')
    plt.tight_layout()
    plt.show()
