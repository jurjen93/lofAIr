# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 10:13:39 2025

@author: ethan
"""
import numpy as np
from argparse import ArgumentParser
import torch
from torchvision.transforms import v2

def findrms(mIn,maskSup=1e-7):
    """
    find the rms of an array, from Cycil Tasse/kMS
    """
    m=mIn[np.abs(mIn)>maskSup]
    rmsold=np.std(m)
    diff=1e-1
    cut=3.
    med=np.median(m)
    for i in range(10):
        ind=np.where(np.abs(m-med)<rmsold*cut)[0]
        rms=np.std(m[ind])
        if np.abs((rms-rmsold)/rmsold)<diff: break
        rmsold=rms
    return rms

def clip_rms(image_data: np.ndarray, clip_rms = 3):
    rms = findrms(image_data)
    image_data[image_data < clip_rms*rms] = 0
    
    return image_data

def augmentations(image_data: np.ndarray):
    # Random flipping, and rotation
    transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticleFlip(p=0.5),
    v2.RandomRotation(180)])
    
    # Performing augmentations on images
    image_data = transforms(image_data)
    
    return image_data

def find_ps(im6: np.ndarray, im03: np.ndarray, rms_cut = 5e-4):
    # TODO: implement beam size, implement function into the code
    """
    Parameters
    ----------
    im6 : 6" resolution image fits file
    im03 : 0.3" resolution image fits file
    rms_cut : TYPE, optional
        DESCRIPTION. The default is 5e-4.

    Returns
    -------
    PS 

    """
    im_6_max = np.nanmax(im6)
    im_03_max = np.nanmax(im03)
    diff_max = im_6_max-im_03_max
    
    if diff_max > diff_max:
        ps_lab = False
    else:
        ps_lab = True
    
    return ps_lab

def parse_args():
    """
    Command line argument parser
    :return: parsed arguments
    """
    parser = ArgumentParser(description='Preprocessing files for GenAI model')
    parser.add_argument('filename', help='Path to filename of image', type=str)
    parser.add_argument('rms_cut', help='Value for defining point sources from non-point sources')
    return parser.parse_args()

def main():
    """ Main function"""
    args = parse_args()
    im_clip = clip_rms(args.filename)
    im_final = augmentations(im_clip)

if __name__ == '__main__':
    main()
    