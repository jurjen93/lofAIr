# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 08:42:56 2026

@author: ethan
"""

import numpy as np
from argparse import ArgumentParser
import astropy.io.fits as fits 
from astropy.wcs import WCS
import glob
import shutil

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

def parse_args():
    """
    Command line argument parser
    :return: parsed arguments
    """
    parser = ArgumentParser(description='Preprocessing files for GenAI model')
    parser.add_argument('folder', help='Path to filename of image at 6"', type=str)
    return parser.parse_args()

def main():
    """ Main function"""
    # TODO: make it so it automatically moves the files
    args = parse_args()
    
    file_list = glob.glob(str(f"{args.folder}/*")) # to loop over all files
    for filename in file_list:  
        im = fits.open(filename)
        im_data = im[0].data   
        rms = findrms(im_data)
        im_data = im_data/ rms
        im_data[im_data<5] = 0
        if np.sum(im_data) == 0:
            new_path = filename.replace("Non_point_source", "Insufficient_rms")
            new_path = new_path.replace("Point_source", "Insufficient_rms")
            shutil.move(filename, new_path)
            
            path_06 = filename.replace("03resolution_output", "6resolution_output")
            path_06 = path_06.replace("03arcs", "6arcs")
            path_06_new = path_06.replace("Non_point_source", "Insufficient_rms")
            path_06_new = path_06_new.replace("Point_source", "Insufficient_rms")
            shutil.move(path_06, path_06_new)

if __name__ == '__main__':
    main()