# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 09:43:18 2026

@author: ethan
"""

from astropy.io import fits
from argparse import ArgumentParser
import numpy as np
import glob
from random import shuffle

def cube_maker(res, pixels):
    # Creating list for the names for training and testing set
    train_list = []
    val_list = []
    
    # HR is currently not split 
    # TODO: split this as well?
    file_list = glob.glob(f"lotss_hr_images_{res}_arcs/*.fits")
    hr_len = len(file_list)
    hr_train_split = int(np.round(0.85*hr_len))
    # Extending lists
    train_list.extend(file_list[0:hr_train_split])
    val_list.extend(file_list[int(hr_train_split+1):])
    
    # All current point sources
    file_list_PS = []
    file_list_PS.extend(glob.glob(f"{res}resolution_output_64/Point_source/*.fits"))
    file_list_PS.extend(glob.glob(f"output_EDFN/{res}resolution_output/Point_source/*.fits"))
    file_list_PS.extend(glob.glob(f"output_Lockman/{res}resolution_output/Point_source/*.fits"))
    file_list_PS.extend(glob.glob(f"output_Bootes/{res}resolution_output/Point_source/*.fits"))
    shuffle(file_list_PS) # To have all surveys in both train and validate set
    ps_len = len(file_list_PS)
    ps_train_split = int(np.round(0.85*ps_len))
    # Extending lists
    train_list.extend(file_list_PS[0:ps_train_split])
    val_list.extend(file_list_PS[int(ps_train_split+1):])
    
    # All current non-point sources
    file_list_NPS = []
    file_list_NPS.extend(glob.glob(f"{res}resolution_output_64/Non_point_source/*.fits"))
    file_list_NPS.extend(glob.glob(f"output_EDFN/{res}resolution_output/Non_point_source/*.fits"))
    file_list_NPS.extend(glob.glob(f"output_Lockman/{res}resolution_output/Non_point_source/*.fits"))
    file_list_NPS.extend(glob.glob(f"output_Bootes/{res}resolution_output/Non_point_source/*.fits"))
    shuffle(file_list_NPS) # To have all surveys in both train and validate set
    nps_len = len(file_list_NPS)
    nps_train_split = int(np.round(0.85*nps_len))
    # Extending lists
    train_list.extend(file_list_NPS[0:nps_train_split])
    val_list.extend(file_list_NPS[int(nps_train_split+1):])
    
    # Shuffle to have different surveys close to each other
    shuffle(train_list)
    shuffle(val_list)
    
    print(train_list[:10])
    
    # Get the depth for cubes
    depth_train = int(len(train_list))
    depth_val = int(len(val_list))
    
    # Make empty cubes for training and testing
    cube_training = np.zeros((depth_train, pixels, pixels))
    cube_validate = np.zeros((depth_val, pixels, pixels))

    i = 0
    for filename in train_list: 
        hdu = fits.open(filename) 
        data = hdu[0].data[:,:]
        cube_training[i,:,:] = data
        i+=1
    
    hdu_new_train = fits.PrimaryHDU(cube_training)
    hdu_new_train.writeto(f'\net\vdesk\data2\WoestE\cube_train_{res}_arcs.fits', overwrite = True)
    
    i = 0
    for filename in val_list: 
        hdu = fits.open(filename) 
        data = hdu[0].data[:,:]
        cube_validate[i,:,:] = data
        i+=1
        
    hdu_new_val = fits.PrimaryHDU(cube_validate)
    hdu_new_val.writeto(f'\net\vdesk\data2\WoestE\cube_val_{res}_arcs.fits', overwrite = True)

def parse_args():
    """
    Command line argument parser
    :return: parsed arguments
    """
    parser = ArgumentParser(description='Preprocessing files for GenAI model')
    parser.add_argument('res', help='resolution 6" or 0.3" arcseconds')
    parser.add_argument('pixels', help='Number of pixels to include in the image dimensions of the cube', type=str)
    return parser.parse_args()

def main():
    """ Main function"""
    args = parse_args()
    cube_maker(int(args.res), int(args.pixels))

if __name__ == '__main__':
    main()