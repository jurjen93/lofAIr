# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 11:52:00 2025

@author: ethan
"""

import numpy as np
import astropy.io.fits as fits 
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.coordinates import SkyCoord
from astropy.visualization import astropy_mpl_style, ImageNormalize, ZScaleInterval, PercentileInterval, AsinhStretch, SqrtStretch, SquaredStretch
from argparse import ArgumentParser

def Imgs_import(filename6):
    # 6 arcsecond resolution
    im_6_arcs = fits.open(filename6)
    im_6_arcs_data = im_6_arcs[0].data[0,0,:,:]
    im_6_arcs_header = im_6_arcs[0].header
    wcs_6_arcs = WCS(im_6_arcs_header, naxis = 2)
    
    # 0.3 arcsecond resolution
    # path_03 = path03
    path_03 = "03resolution/"
    all_im_03_arcs = []
    for i in range(30):
        im_03 = fits.open(path_03 + f"facet_{i}.fits")
        all_im_03_arcs.append((im_03[0].header, 
            im_03[0].data,
            WCS(im_03[0].header, naxis=2)
        ))
    return im_6_arcs_data, im_6_arcs_header, wcs_6_arcs , all_im_03_arcs
    
def Cat_import(filename):
    catelog = fits.open(filename)
    # Improvement! -> automated the recognition of columns and splitting to work for any combination
    
    # Getting all colums from catelog 
    # E columns are error columns
    # Different database but label descriptions seem the same https://heasarc.gsfc.nasa.gov/w3browse/all/nvss.html 
    SOURCE_ID = catelog[1].data['Source_ID'] 
    CAT_ID = catelog[1].data['Cat_id']
    RA = catelog[1].data['RA'] # deg
    E_RA = catelog[1].data['E_RA'] # deg
    DEC = catelog[1].data['DEC'] # deg
    E_DEC = catelog[1].data['E_DEC'] # deg
    TOTAL_FLUX = catelog[1].data['Total_flux'] # Jy
    E_TOTAL_FLUX = catelog[1].data['E_Total_flux'] # Jy
    PEAK_FLUX = catelog[1].data['Peak_flux'] # beam-1 Jy
    E_PEAK_FLUX = catelog[1].data['E_Peak_flux'] # beam-1 Jy
    MAJ = catelog[1].data['Maj'] # deg
    E_MAJ = catelog[1].data['E_Maj'] # deg
    MIN = catelog[1].data['Min'] # deg
    E_Min = catelog[1].data['E_Min'] # deg
    PA = catelog[1].data['PA'] # deg
    E_PA = catelog[1].data['E_PA '] # deg
    S_CODE = catelog[1].data['S_Code']
    ISL_RMS = catelog[1].data['Isl_rms']
    
    return SOURCE_ID, CAT_ID, RA, DEC # Probably could also make this more efficient

def Take_snippet(source_idx, r, RA, DEC, CAT_ID, SOURCE_ID, all_im_03_arcs, im_6_arcs_data, im_6_arcs_header, wcs_6_arcs):
    """
    Creating a cutout in both 6 and 0.3 arcseconds. As well as creating an image
    inputs:
        source_idx = index used to gain results from catelog
        r = number of pixels in 6 arcseconds (rxr cutout)
        RA, DEC, CAT_ID, SOURCE_ID are taken from catelog
        rest is from loading in images
        
    outputs:
        fits file 6 arcseconds cutout
        fits file 0.3 arcseconds cutout
        png file comparing the two
    """
    # Setting the basics
    scale = 15 # scaling between 6" and 0.3" -> could also be defined outside function
    c_RA, c_DEC = RA[source_idx], DEC[source_idx] 
    coord = SkyCoord(c_RA, c_DEC, unit="deg")

    # Finding correct facet for 0.3"
    cat_id = CAT_ID[source_idx]
    facet = int(str(cat_id )[:2].replace("_", " "))
    im_03_arcs_header, im_03_arcs_data, wcs_03_arcs = all_im_03_arcs[facet]

    # 6" cutout
    cutout_6_arcs = Cutout2D(im_6_arcs_data, coord, r, wcs_6_arcs)
    cutout_6_image = cutout_6_arcs.data
    cutout_6_header = cutout_6_arcs.wcs.to_header()

    # 0.3" cutout
    cutout_03_arcs = Cutout2D(im_03_arcs_data, coord, r*scale, wcs_03_arcs)
    cutout_03_image = cutout_03_arcs.data
    cutout_03_header = cutout_03_arcs.wcs.to_header()
    
    # Saving files
    path_6_output = '/net/vdesk/data2/WoestE/6resolution_output/'
    hdu_6_arc = fits.PrimaryHDU(header=cutout_6_header, data=cutout_6_image)
    hdu_6_arc.writeto(str(path_6_output)+str(SOURCE_ID[source_idx])+"_Rad"+str(r)+"_6arcs.fits", overwrite=True)

    path_03_output = '/net/vdesk/data2/WoestE/03resolution_output/'
    hdu_03_arc = fits.PrimaryHDU(header=cutout_03_header, data=cutout_03_image)
    hdu_03_arc.writeto(str(path_03_output)+str(SOURCE_ID[source_idx])+"_Rad"+str(r)+"_03arcs.fits", overwrite=True)
    
    # Finding the noise levels
    rms_6_imp = findrms(cutout_6_image)
    rms_03_imp = findrms(cutout_03_image)

    # Defining upper and lower limits in terms of s/n ratio in plots
    vmin = 5
    vmax = 50
    
    # Plotting
    ax1 = plt.subplot((121), projection = WCS(cutout_6_header, naxis = 2))
    plt.imshow(np.abs(cutout_6_image), cmap = 'inferno', norm = colors.LogNorm(vmin * rms_6_imp, vmax * rms_6_imp))
    # plt.colorbar() -> they are different right
    plt.grid(color='white', linestyle = 'dashed')    

    ax2 = plt.subplot((122), projection = WCS(cutout_03_header, naxis = 2),)
    plt.imshow(np.abs(cutout_03_image), cmap = 'inferno', norm = colors.LogNorm(vmin * rms_03_imp, vmax * rms_03_imp))
    # plt.colorbar()
    plt.grid(color='white', linestyle = 'dashed')
    path_im = '/net/vdesk/data2/WoestE/pngs/'
    plt.savefig(path_im+str(SOURCE_ID[source_idx])+"_Rad"+str(r)+".png")
    plt.clf()
    
def parse_args():
    """
    Command line argument parser
    :return: parsed arguments
    """
    parser = ArgumentParser(description='Crop fits files in multiple resolutions')
    parser.add_argument('filename6', help='fits input 6 arcseconds', type=str)
    # parser.add_argument('path', help='path to input files 0.3 arcseconds', type=str) 
    # Wasn't recognised in main so took it out for now but will import again later for automation purporses
    parser.add_argument('cat', help='catelog file', type=str)
    return parser.parse_args()

def findrms(mIn,maskSup=1e-7):
    """
    find the rms of an array, from Cycil Tasse/kMS
    """
    m=mIn[np.abs(mIn)>maskSup]
    rmsold=np.std(m)
    diff=1e-1
    cut=3.
    bins=np.arange(np.min(m),np.max(m),(np.max(m)-np.min(m))/30.)
    med=np.median(m)
    for i in range(10):
        ind=np.where(np.abs(m-med)<rmsold*cut)[0]
        rms=np.std(m[ind])
        if np.abs((rms-rmsold)/rmsold)<diff: break
        rmsold=rms
    return rms


def main():
    """ Main function"""
    args = parse_args()
    im_6_arcs_data, im_6_arcs_header, wcs_6_arcs , all_im_03_arcs = Imgs_import(args.filename6)
    SOURCE_ID, CAT_ID, RA, DEC = Cat_import(args.cat)
    for i in range(len(RA)): # Need to automatically find length again!
        Take_snippet(i, 2048, RA, DEC, CAT_ID, SOURCE_ID, all_im_03_arcs, im_6_arcs_data, im_6_arcs_header, wcs_6_arcs)

if __name__ == '__main__':
    main()