# Customary Imports:
import numpy as np
import matplotlib.pyplot as plt
import math
import string
import pandas as pd
import skimage
import os
import re
import datetime
import scipy
from skimage import exposure
import seaborn as sns
import random
import shutil
import PIL
import imageio
import copy
from pathlib import Path
import utils.remove_pad_save_to_new_dir

##################################################################################################################################
'''
REMOVING PADDING FROM IMAGE AND SAVING IN OUTPUT DIR:
'''
def remove_padding(img):
    # Removing Image Border (if it exists)
    xmin=0
    xmax=img.shape[0]-1
    ymin=0
    ymax=img.shape[1]-1
    for xmin in range(img.shape[0]):
        if np.sum(img[xmin,:,0]) > 0.01:
            break
    for xmax in range(img.shape[0]-1, 0, -1):
        if np.sum(img[xmax,:,0]) > 0.01:
            break
    for ymin in range(img.shape[1]):
        if np.sum(img[:,ymin,0]) > 0.01:
            break
    for ymax in range(img.shape[1]-1, 0, -1):
        if np.sum(img[:,ymax,0]) > 0.01:
            break
    no_pad = img[xmin:xmax,ymin:ymax+1,...]
    return no_pad

def remove_pad_save_to_new_dir(input_dir, output_dir, file_format='.png', num_images=1):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    file_list = [file for file in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, file))]
    for file in file_list[:num_images]:
        # Load Image
        filepath = os.path.join(input_dir, file)
        if filepath.endswith('.npy'):
            array = np.load(filepath)
        else:
            array = imageio.imread(filepath)
            array = np.array(array)
        new_array = remove_padding(array.astype(np.float32))
        new_filepath = Path(os.path.join(output_dir, file))
        new_filepath = Path(os.path.abspath(new_filepath.with_suffix('')) + file_format)
        if file_format == '.npy':
            np.save(new_filepath, new_array, allow_pickle=True, fix_imports=True)
        elif file_format == '.tif' or file_format == '.tiff':
            imageio.imwrite(new_filepath, new_array, file_format)
        else:
            new_array = exposure.rescale_intensity(new_array, in_range='image', out_range=(0.0,255.0)).astype(np.uint8)
            imageio.imwrite(new_filepath, new_array, file_format)