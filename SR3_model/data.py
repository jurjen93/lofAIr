import random
from PIL import Image
from os import listdir
from os.path import join
import numpy as np
from skimage.transform import resize

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from preprocessing import findrms, clip_rms, rescale
from astropy.io import fits

class TrainDataset(Dataset):
    def __init__(self, img_path, lbl_path, crop_size):
        # Importing data
        self.img_path = img_path
        self.lbl_path = lbl_path
        
        self.scale_factor = 4 # For 4x upscaling
        self.crop_size = crop_size 
        self.tensor = transforms.ToTensor()

        # Getting cubes
        img_data = fits.open(self.img_path)[0].data
        img_data = img_data.astype(np.float32)
        self.img_cube = self.tensor(img_data) 
        lbl_data = fits.open(self.lbl_path)[0].data
        lbl_data = resize(lbl_data, [len(self.img_cube[0,:,0]), self.crop_size*self.scale_factor, self.crop_size*self.scale_factor], anti_aliasing=True)
        lbl_data = lbl_data.astype(np.float32)
        self.lbl_cube = self.tensor(lbl_data)
        
    def __len__(self):
        return len(self.img_cube[0,:,0]) 

    def __getitem__(self, idx):
        img = self.img_cube[:, idx, :]
        lbl = self.lbl_cube[:, idx, :]
        
        # random crop
        params = transforms.RandomCrop(self.crop_size).get_params(img, (self.crop_size, self.crop_size)) 
        img = transforms.functional.crop(img, *params) #
        lbl = transforms.functional.crop(lbl, *[self.scale_factor*p for p in params])

        # Transoformations need a 4d array?
        img = img.unsqueeze(0).unsqueeze(0)
        lbl = lbl.unsqueeze(0).unsqueeze(0)

       # random flip
        if random.random() < 0.5: 
            img = torch.flip(img, [2])
            lbl = torch.flip(lbl, [2])
        # random rotation
        angle = float(random.randint(0, 359))
        img = transforms.functional.rotate(img, angle)
        lbl = transforms.functional.rotate(lbl, angle)

        # Returning to dimensions of 2D image
        img = img.squeeze()
        lbl = lbl.squeeze()
        
        # clip_rms
        img = clip_rms(img, clip_rms = 3)
        lbl = clip_rms(lbl, clip_rms = 3)
        
        # Adding empty layers as replacement for expected 3 colors
        empty_layer_lr = torch.zeros(self.crop_size, self.crop_size)
        empty_layer_hr = torch.zeros(self.crop_size*self.scale_factor, self.crop_size*self.scale_factor)
        img = torch.stack([img, empty_layer_lr, empty_layer_lr])
        lbl = torch.stack([lbl, empty_layer_hr, empty_layer_hr])

        return img, lbl
    
class EvalDataset(Dataset):
    def __init__(self, img_path, lbl_path, crop_size=None):
        # Importing data
        self.img_path = img_path
        self.lbl_path = lbl_path
        
        self.scale_factor = 4 # For 4x upscaling
        self.crop_size = crop_size
        self.tensor = transforms.ToTensor()

        # Getting cube
        img_data = fits.open(self.img_path)[0].data
        img_data = img_data.astype(np.float32)
        self.img_cube = self.tensor(img_data) 
        lbl_data = fits.open(self.lbl_path)[0].data
        plt.imshow(lbl_data[0,:,:])
        lbl_data = resize(lbl_data, [len(self.img_cube[0,:,0]), self.crop_size*self.scale_factor, self.crop_size*self.scale_factor], anti_aliasing=True)
        plt.imshow(lbl_data[0,:,:])
        lbl_data = lbl_data.astype(np.float32)
        self.lbl_cube = self.tensor(lbl_data)

    def __len__(self):
        return len(self.img_cube[0, :, 0]) 

    def __getitem__(self, idx):
        img = self.img_cube[:, idx, :]
        lbl = self.lbl_cube[:, idx, :]

        # clip_rms
        img = clip_rms(img, clip_rms = 3)
        lbl = clip_rms(lbl, clip_rms = 3)
        
        # crop
        params = transforms.RandomCrop(self.crop_size).get_params(img, (self.crop_size, self.crop_size)) 
        img = transforms.functional.crop(img, *params) #
        lbl = transforms.functional.crop(lbl, *[self.scale_factor*p for p in params])

        # Adding empty layers as replacement for expected 3 colors
        empty_layer_lr = torch.zeros(self.crop_size, self.crop_size)
        empty_layer_hr = torch.zeros(self.crop_size*self.scale_factor,self.crop_size*self.scale_factor)
        img = torch.stack([img, empty_layer_lr, empty_layer_lr])
        lbl = torch.stack([lbl, empty_layer_hr, empty_layer_hr])

        return img, lbl

class Dataset(TrainDataset):
    def __init__(self,
                img_path,
                lbl_path,
                crop_size):
        super().__init__(
            img_path,
            lbl_path,
            crop_size
        )
"""
class Flickr2KDataset(TrainDataset):
    def __init__(self,
                img_path,
                lbl_path,
                crop_size):
        super().__init__(
            img_path,
            lbl_path,
            crop_size
        )
"""
class TrainDataset(Dataset):
    def __init__(self, lr_path, hr_path, crop_size):
        super().__init__(lr_path, hr_path, crop_size)
        self.datatrain = Dataset(lr_path, hr_path, crop_size)
        self.datatrain_len = len(self.datatrain)
        self.total_imgs = self.datatrain_len
    
    def __len__(self):
        return self.total_imgs
    
    def __getitem__(self, idx):
        return self.datatrain.__getitem__(idx)

class ValDataset(EvalDataset):
    def __init__(self,
                img_path,
                lbl_path,
                crop_size):
        super().__init__(
            img_path,
            lbl_path,
            crop_size
        )

class TestDataset(EvalDataset):
    def __init__(self,
                img_path,
                lbl_path,
                crop_size):
        super().__init__(
            img_path,
            lbl_path,
            crop_size
        )
