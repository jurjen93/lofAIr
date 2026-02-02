import random
from PIL import Image
from os import listdir
from os.path import join
import numpy as np
from skimage.transform import resize

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from preprocessing import findrms, clip_rms, rescale
from astropy.io import fits

class TrainDataset(Dataset):
    def __init__(self, img_path, lbl_path, crop_size):
        # TODO: switch this to work with cube files
        # Importing data
        # self.img_names = sorted([name for name in listdir(img_path)])
        # self.lbl_names = sorted([name for name in listdir(lbl_path)])
        
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
        lbl_data = resize(lbl_data, [len(self.img_cube[0,:,0]),256, 256])
        lbl_data = lbl_data.astype(np.float32)
        self.lbl_cube = self.tensor(lbl_data)
        
    def __len__(self):
        return len(self.img_cube[0,:,0]) 

    def __getitem__(self, idx):
        img = self.img_cube[:, idx, :]
        lbl = self.lbl_cube[:, idx, :]

        # clip_rms
        img = clip_rms(img, clip_rms = 3)
        lbl = clip_rms(lbl, clip_rms = 3)

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

        # Returning to dimensions of 64x64 --> check if this is correct or not
        img = img.squeeze()
        lbl = lbl.squeeze()

        return img, lbl
    
class EvalDataset(Dataset):
    def __init__(self, img_path, lbl_path, crop_size=None):
        # TODO: switch this to work with cube files
        # Importing data
        # self.img_names = sorted([name for name in listdir(img_path)])
        # self.lbl_names = sorted([name for name in listdir(lbl_path)])
        
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
        lbl_data = resize(lbl_data, [len(self.img_cube[0,:,0]),256, 256])
        lbl_data = lbl_data.astype(np.float32)
        self.lbl_cube = self.tensor(lbl_data)

    def __len__(self):
        return len(self.img_cube[0, :, 0]) 

    def __getitem__(self, idx):
        img = img_cube[:, idx, :]
        lbl = lbl_cube[:, idx, :]

        # crop
        if self.crop_size != None:
            img = transforms.CenterCrop(self.crop_size)(img) # Cropping to fit size
            lbl = transforms.CenterCrop(self.scale_factor*self.crop_size)(lbl)

        # clip_rms
        img = clip_rms(img, clip_rms = 3)
        lbl = clip_rms(img, clip_rms = 3)

        return img, lbl

class DIV2kDataset(TrainDataset):
    # No clue what this does?
    def __init__(self,
                img_path,
                lbl_path,
                crop_size):
        super().__init__(
            img_path,
            lbl_path,
            crop_size
        )

class Flickr2KDataset(TrainDataset):
    # No clue what this does?
    def __init__(self,
                img_path,
                lbl_path,
                crop_size):
        super().__init__(
            img_path,
            lbl_path,
            crop_size
        )

class DF2KTrainDataset(Dataset):
    def __init__(self, div2k_lr_path, div2k_hr_path, flickr2k_lr_path, flickr2k_hr_path, crop_size):
        super().__init__()
        self.div2k = DIV2kDataset(div2k_lr_path, div2k_hr_path, crop_size)
        self.flickr2k = Flickr2KDataset(flickr2k_lr_path, flickr2k_hr_path, crop_size)
        self.div2k_len = len(self.div2k)
        self.flickr2k_len = len(self.flickr2k)
        self.total_imgs = self.div2k_len+self.flickr2k_len # This right now is just training + testing set, what are these names?
    
    def __len__(self):
        return self.total_imgs
    
    def __getitem__(self, idx):
        if idx < self.div2k_len:
            return self.div2k.__getitem__(idx)
        else:
            return self.flickr2k.__getitem__(idx-self.div2k_len)

class DIV2KValDataset(EvalDataset):
    # Isn't this the same as before?
    def __init__(self,
                img_path,
                lbl_path,
                crop_size):
        super().__init__(
            img_path,
            lbl_path,
            crop_size
        )

class Flickr2KTestDataset(EvalDataset):
    def __init__(self,
                img_path,
                lbl_path,
                crop_size):
        super().__init__(
            img_path,
            lbl_path,
            crop_size
        )
