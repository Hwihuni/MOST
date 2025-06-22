from os.path import splitext
from os import listdir
import numpy as np 
import os
import scipy.io
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import fastmri

class BasicDataset(Dataset):
    def __init__(self, args,mode,memory = False,memory_task = None):
        self.task = args.task
        if memory:
            path_image = f'./pickle_buffer/{args.name}/pkl_memory_{memory_task}_{mode}.pklv4'
            path_target = f'./pickle_buffer/{args.name}/pkl_memory_{memory_task}_{mode}_fimg.pklv4'
            self.task = memory_task
        else:        
            path_image = f'./pickle_train/pkl_{args.task}_{mode}.pklv4'
        self.target = self.load_pkls(path_image)
        self.target = [np.transpose(image, [2, 0, 1]) for image in self.target]
        self.real_target = self.load_pkls(path_target)
        self.real_target = [np.transpose(image, [2, 0, 1]) for image in self.real_target]
        self.args = args
        num_cols = self.target[0].shape[1]
        num_rows = self.target[0].shape[2]
        center_fraction = args.center_fraction
        acceleration = args.acceleration  
        num_low_freqs = int(round(num_cols * center_fraction))
        self.num_low_freqs = num_low_freqs
        # create the mask
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True

        # determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
            num_low_freqs * acceleration - num_cols
        )
        
        offset = 0
        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = True
        
        # reshape the mask
        mask_shape = [1,num_cols,1]
        self.mask2 = mask.reshape(*mask_shape)
        self.mask = np.repeat(self.mask2,num_rows,axis =2)
        logging.info(f'Creating dataset with {len(self.target)} examples')
        
    def load_pkls( self,path):
        assert os.path.isfile(path), path
        images = []
        with open(path, "rb") as f:
            images += pickle.load(f)
        assert len(images) > 0, path
        return images
        
    @classmethod
    def undersample(cls, x,mask):
        x_k = fft2c(x)
        masked_kspace_sudo = mask*x_k +0.0
        return masked_kspace_sudo

    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, i):
        masked_kspace_sudo = self.undersample(self.target[i],self.mask)
        image_undersample = np.abs(ifft2c(masked_kspace_sudo))
        return image_undersample.astype('float32'), np.stack((np.real(masked_kspace_sudo),np.imag(masked_kspace_sudo)),3).astype('float32'), self.mask, self.real_target[i].astype('float32'), self.real_target[i].astype('float32')


def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(x,axes = (-2,-1)),norm='ortho'),axes = (-2,-1))

def fft2c(x):
    return np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(x,axes = (-2,-1)),norm='ortho'),axes = (-2,-1))
