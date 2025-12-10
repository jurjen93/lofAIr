# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 13:19:16 2025

@author: ethan
"""
import glob
import shutil
 
def main():
    """ Changing all paths"""
    path_in = "/net/vdesk/data2/WoestE/surveys/*/*/selfcal/*/*.fits"
    file_list = glob.glob(path_in) # to loop over all files
    
    print(len(file_list))
    for filename in file_list:        
        old_path = filename[:-25]
        source_name = filename[55:-25]
        
        new_path = f"/net/vdesk/data2/WoestE/lotss_hr_images/{source_name}.fits"
        
        # os gave errors, shutil.move does the same but gave no errors
        shutil.move(filename, new_path)
    
if __name__ == '__main__':
    main()