# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 11:02:41 2025

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

def imgs_import(filename6, filename03):
    # 6 arcsecond resolution
    im_6_arcs = fits.open(filename6)
    im_6_arcs_data = im_6_arcs[0].data[0,0,:,:]
    im_6_arcs_header = im_6_arcs[0].header
    wcs_6_arcs = WCS(im_6_arcs_header, naxis = 2)
    
    # 0.3 arcsecond resolution
    im_03_arcs = fits.open(filename03)
    im_03_arcs_data = im_03_arcs[0].data
    im_03_arcs_header = im_03_arcs[0].header
    wcs_03_arcs = WCS(im_03_arcs_header, naxis = 2)
    
    return im_6_arcs_data, im_6_arcs_header, wcs_6_arcs, im_03_arcs_data, im_03_arcs_header, wcs_03_arcs 
    
def cat_import(filename):
    catelog = fits.open(filename)
    # Improvement! -> automated the recognition of columns and splitting to work for any combination
    
    # Getting all colums from catelog 
    # E columns are error columns
    # Different database but label descriptions seem the same https://heasarc.gsfc.nasa.gov/w3browse/all/nvss.html 
    source_name = catelog[1].data['Source_Name_LOTSS']
    ra = catelog[1].data['RA'] # deg
    dec = catelog[1].data['DEC'] # deg
    
    return  source_name, ra, dec, # Probably could also make this more efficient

def take_snippet(source_idx, r, ra, dec, source_name, im_03_arcs_data, im_03_arcs_header, wcs_03_arcs, im_6_arcs_data, im_6_arcs_header, wcs_6_arcs):
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
    c_RA, c_DEC = ra[source_idx], dec[source_idx]
    coord = SkyCoord(c_RA, c_DEC, unit="deg")

    # 6" cutout
    cutout_6_arcs = Cutout2D(im_6_arcs_data, coord, r, wcs_6_arcs)
    cutout_6_image = cutout_6_arcs.data
    cutout_6_header = cutout_6_arcs.wcs.to_header()

    # 0.3" cutout
    cutout_03_arcs = Cutout2D(im_03_arcs_data, coord, r*scale, wcs_03_arcs)
    cutout_03_image = cutout_03_arcs.data
    cutout_03_header = cutout_03_arcs.wcs.to_header()
    
    # Finding the noise levels
    rms_6_imp = findrms(cutout_6_image)
    rms_03_imp = findrms(cutout_03_image)
       
    # Finding peak 
    im_6_max = np.nanmax(cutout_6_image)
    im_03_max = np.nanmax(cutout_03_image)
    diff_max = im_6_max-im_03_max
   
    cut_off_ps = 5e-4
    
    # Defining upper and lower limits in terms of s/n ratio in plots
    if im_6_max >= rms_6_imp:
       vmax_6 = int(np.round(im_6_max/rms_6_imp))
       vmin_6 = 3
    else: 
       vmax_6 =  int(np.round(im_6_max/rms_6_imp))
       vmin_6 = 1
   
    if im_03_max >= rms_03_imp:
       vmax_03 = int(np.round(im_03_max/rms_03_imp))
       vmin_03 = 3
    else: 
       vmax_03 =  int(np.round(im_03_max/rms_03_imp))
       vmin_03 = 1
       
    # Now moving onto finding splitting point and non point sources
    if diff_max > cut_off_ps:
        path_6_output = '/net/vdesk/data2/WoestE/output_Lockman/6resolution_output/Non_point_source/'
        path_03_output = '/net/vdesk/data2/WoestE/output_Lockman/03resolution_output/Non_point_source/'
    else:
        path_6_output = '/net/vdesk/data2/WoestE/output_Lockman/6resolution_output/Point_source/'
        path_03_output = '/net/vdesk/data2/WoestE/output_Lockman/03resolution_output/Point_source/'
           
    # Saving files
    hdu_6_arc = fits.PrimaryHDU(header=cutout_6_header, data=cutout_6_image)
    hdu_6_arc.writeto(str(path_6_output)+str(source_name[source_idx])+"_Rad"+str(r)+"_6arcs.fits", overwrite=True)

    hdu_03_arc = fits.PrimaryHDU(header=cutout_03_header, data=cutout_03_image)
    hdu_03_arc.writeto(str(path_03_output)+str(source_name[source_idx])+"_Rad"+str(r)+"_03arcs.fits", overwrite=True)
       
    # Plotting
    ax1 = plt.subplot((121), projection = WCS(cutout_6_header, naxis = 2))
    plt.imshow(np.abs(cutout_6_image), cmap = 'inferno', norm = colors.LogNorm(vmin_6 * rms_6_imp, vmax_6 * rms_6_imp))
    # plt.colorbar() -> they are different right  

    ax2 = plt.subplot((122), projection = WCS(cutout_03_header, naxis = 2),)
    ax2.coords['pos.eq.dec'].set_ticks_visible(False)
    ax2.coords['pos.eq.dec'].set_ticklabel_visible(False)
    ax2.coords['pos.eq.dec'].set_axislabel('')   
    plt.imshow(np.abs(cutout_03_image), cmap = 'inferno', norm = colors.LogNorm(vmin_03 * rms_03_imp, vmax_03 * rms_03_imp))
    # plt.colorbar()
    path_im = '/net/vdesk/data2/WoestE/output_Lockman/pngs/'
    plt.savefig(path_im+str(source_name[source_idx])+"_Rad"+str(r)+".png")
    plt.clf()
    
def parse_args():
    """
    Command line argument parser
    :return: parsed arguments
    """
    parser = ArgumentParser(description='Crop FITS files at multiple resolutions')
    parser.add_argument('filename6', help='Path to 6 arcsecond FITS image', type=str)
    parser.add_argument('filename03', help='Path to 0.3 arcsecond FITS image', type=str)
    # parser.add_argument('path', help='path to input files 0.3 arcseconds', type=str) 
    # Wasn't recognised in main so took it out for now but will import again later for automation purporses
    parser.add_argument('cat', help='Path to catalogue', type=str)
    return parser.parse_args()

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


def main():
    """ Main function"""
    args = parse_args()
    im_6_arcs_data, im_6_arcs_header, wcs_6_arcs, im_03_arcs_data, im_03_arcs_header, wcs_03_arcs = imgs_import(args.filename6, args.filename03)
    source_name, ra, dec = cat_import(args.cat)
    for i in range(len(ra)): # Need to automatically find length again!
        take_snippet(i, 64, ra, dec, source_name, im_03_arcs_data, im_03_arcs_header, wcs_03_arcs, im_6_arcs_data, im_6_arcs_header, wcs_6_arcs)

if __name__ == '__main__':
    main()