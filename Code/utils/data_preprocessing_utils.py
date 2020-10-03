# Customary Imports:
import tensorflow as tf
assert '2.' in tf.__version__  # make sure you're using tf 2.0
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
import scipy
from skimage import exposure
import h5py
import random
import shutil
import PIL
import imageio
from pathlib import Path
from tensorflow.keras import backend as K
from PIL import Image
import utils.data_preprocessing_utils
##################################################################################################################################
'''
DATA PREPROCESSING UTILS:
'''
##################################################################################################################################
# Converting MAP Files:
def convert_MAP(directory, output_directory, min_shape, file_format = '.npy', search_keys = None, 
                dtype = np.float32, use_avg = True, remove_noisy = False):
    '''
    This program loops through given raw_data directory
    and converts .mat files to .npy files
    '''
    new_dir = os.path.join(os.getcwd(), output_directory)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    else:
        shutil.rmtree(new_dir)
        os.mkdir(new_dir)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".mat"): 
            #print(os.path.join(directory, filename))
            filepath = os.path.join(directory, filename)
            array_dict = {}
            try:
                array_dict = h5py.File(filepath, 'r')
            except:
                array_dict = sio.loadmat(filepath)
            # As we only need image info from dict (the last key) we do this
            if search_keys == None:
                search_keys = 'map' # out of struct of .mat files want "map"
                filtered_dict = dict(filter(lambda item: search_keys in item[0], array_dict.items()))
            else:
                filtered_dict = {}
                for i in range(len(search_keys)):
                    search_key = search_keys[i]
                    if search_key in array_dict:
                        filtered_dict[search_key] = np.array(array_dict[search_key], dtype = np.float32)
            if len(filtered_dict) == 0:
                print('No Data to Meet Search Key Requirements: Datapoint Rejected -> ' + filepath)
            else:
                #print(list(array_dict.keys()))
                #print(filtered_dict)
                arrays = []
                for k, v in filtered_dict.items():
                    temp = np.transpose(v.astype(np.float32))
                    # To normalize data between [-1,1], use -> arrays = np.abs(array)/(np.max(np.abs(array))/2) - 1
                    # To normalize data between [0,1], use -> arrays = np.abs(array)/(np.max(np.abs(array)))
                    # To normalize data between [0,255], 
                    #     use -> arrays = (arrays/(np.max(arrays))*255).astype(np.uint8)
                    # Or use exposure.rescale_intensity function
                    #temp = exposure.rescale_intensity(temp, in_range='image', out_range=(0.0,1.0))
                    temp = np.abs(temp)/np.max(np.abs(temp))
                    if temp.shape[0] > temp.shape[1]:
                        temp = np.rot90(temp)
                    arrays.append(temp)
                for i in range(len(arrays)):
                    if len(arrays[i].shape) > 2:
                        #print(arrays[i].shape)
                        arrays[i] = rgb2gray(arrays[i], use_avg = use_avg)
                        
                for i in range(len(arrays)):
                    new_dir_filepath = os.path.join(new_dir, filename.strip('.mat') 
                                                    + '_index'+str(i) + file_format)
                    array = arrays[i]
                    #print(f'MEAN and STD: {np.mean(array)}, {np.std(array)} -> ' + filepath)
                    if array.shape[0] >= min_shape[0] and array.shape[1] >= min_shape[1]:
                        if (np.mean(array) > 0.3 or (np.std(array) < 0.03 and np.mean(array) > 0.03)) and remove_noisy:
                            print('Noisy Image: Datapoint Rejected -> ' + filepath)
                        else:
                            #print(f'MEAN and STD: {np.mean(array)}, {np.std(array)} -> ' + filepath)
                            if file_format == '.npy':
                                np.save(new_dir_filepath, array, allow_pickle=True, fix_imports=True)
                            elif file_format == '.tif' or file_format == '.tiff':
                                imageio.imwrite(new_dir_filepath, array)
                            else:
                                array = exposure.rescale_intensity(array, in_range='image', out_range=(0.0,255.0)).astype(np.uint8)
                                imageio.imwrite(new_dir_filepath, array)
                    else:
                        print('Min Size Not Met: Datapoint Rejected -> ' + filepath)
    return os.path.join(os.getcwd(), output_directory)

def rgb2gray(img, use_avg = False):
    if use_avg:
        output = np.mean(img, axis = 2)
    else:
        output = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    return output
##################################################################################################################################
# Data Cleaning Procedures:
def data_clean_func(image = None, threshold = 0.85, contrast_enhance = True):
    if image is not None:
        #print(len(np.unique(image)))
        #clean_image = image
        '''
        plt.hist(image)
        plt.show()
        '''
        '''
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.show()
        '''
        if threshold is not None:
            image = skimage.img_as_float(image)
            p1, p2 = np.percentile(image, threshold)
            image = exposure.rescale_intensity(image, in_range=(p1, p2), out_range=(0.0,1.0))
        elif threshold == 0.85:
            # Original Thresholding Procedure - DEFAULT
            default_fill = 0.0
            frac_of_high_clip = 1/9
            image[image > threshold] = default_fill
            image[image < frac_of_high_clip*(1.0-threshold)] = default_fill
        '''
        plt.imshow(image, cmap='gray')
        plt.title('After Clipping')
        plt.show()
        '''
        image = scipy.ndimage.median_filter(image, size=(4, 4))
        '''
        plt.imshow(image, cmap='gray')
        plt.title('After Median Filter')
        plt.show()
        '''
        image = skimage.filters.gaussian(image, sigma=0.01, output=None, mode='reflect', preserve_range=True)
        if contrast_enhance:
            ####################################################################
            # Added to ensure negligible loss when converting to int16 
            # within exposure.equalize_adapthist
            #image = exposure.rescale_intensity(image, in_range='image', out_range=(0.0,2**16)).astype(np.uint16)
            # A "Monkey Patch" could possibly be used as a cleaner solution, 
            # but would be more involved than is necessary for my application
            ####################################################################
            image = exposure.equalize_adapthist(image,kernel_size=image.shape[0]//8, clip_limit=0.005, nbins=2**9)
            image = skimage.img_as_float(image)
            p1, p2 = np.percentile(image, (10.0,100.0))
            image = exposure.rescale_intensity(image, in_range=(p1, p2), out_range=(0.0,1.0))
        image = image.astype(np.float64)
        '''
        plt.imshow(image, cmap='gray')
        plt.title('After Local Adapt Hist')
        plt.show()
        '''
        image = scipy.ndimage.median_filter(image, size=(3, 1))
        image = scipy.ndimage.median_filter(image, size=(1, 3))
        image = skimage.filters.gaussian(image, sigma=0.1, output=None, mode='reflect', preserve_range=True)
        image = exposure.rescale_intensity(image, in_range='image', out_range=(0.0,1.0))
        '''
        plt.imshow(image, cmap='gray')
        plt.title('Final Image')
        plt.show()
        '''
        '''
        plt.hist(image)
        plt.show()
        '''
        clean_image = image.astype(np.float32)
    else:
        clean_image = image
    return clean_image

def data_cleaning(input_dir = 'converted_data', output_dir_name = 'cleaned_data',
                  output_file_format ='.npy', threshold = 0.85, contrast_enhance = True, 
                  delete_previous = True):
    '''
     This program seeks to remove some noise from the data
     and make the underlying vessel structure more prominent
     Input: input_dir -> directory that holds data to be cleaned
            output_dir -> directory to hold cleaned data
     Output: None
    '''
    file_list = os.listdir(input_dir)
    clean_dir = os.path.join(os.getcwd(), output_dir_name)
    if not os.path.exists(clean_dir):
        os.mkdir(clean_dir)
    elif delete_previous == True:
        shutil.rmtree(clean_dir)
        os.mkdir(clean_dir)
    for file in file_list:
        filename = os.fsdecode(file)
        filepath = os.path.join(input_dir, filename)
        if filepath.endswith('.npy'):
            array = np.load(filepath).astype(np.float32)
        else:
            array = imageio.imread(filepath).astype(np.float32)
            
        # Defined data clean function above:
        array = data_preprocessing_utils.data_clean_func(array, threshold, contrast_enhance)
    
        new_filepath = os.path.join(clean_dir, filename)
        new_filepath = Path(new_filepath)
        new_filepath = new_filepath.with_suffix('')
        new_filepath = new_filepath.with_suffix(output_file_format)
        
        if output_file_format == '.npy':
            np.save(new_filepath, array, allow_pickle=True, fix_imports=True)
        elif output_file_format == '.tif' or output_file_format == '.tiff':
            imageio.imwrite(new_filepath, array, output_file_format)
        else:
            array = exposure.rescale_intensity(array, in_range='image', out_range=(0.0,255.0)).astype(np.uint8)
            imageio.imwrite(new_filepath, array, output_file_format)
    return  
##################################################################################################################################
# Data Removal / Transfer Procedures:

def transfer_files_except(input_dir, output_dir, exception_list, delete_previous = True):
    file_list = os.listdir(input_dir)
    new_dir = os.path.join(os.getcwd(), output_dir)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    elif delete_previous == True:
        shutil.rmtree(new_dir)
        os.mkdir(new_dir)
    for file in file_list:
        if file not in exception_list and file != '.ipynb_checkpoints':
            filepath = os.path.join(input_dir, file)
            new_filepath = os.path.join(output_dir, file)
            shutil.copyfile(filepath, new_filepath)
    
##################################################################################################################################
# Data Seperation / Validation Split Procedures:
def data_seperation(input_dir, output_dir = 'data', dataset_percentages = (90,5,5), delete_previous = False, 
                    file_format = '.npy', scale = 1.0):
    '''
    Takes numpy array and creates data folder with seperate sections
    for training, validation, and testing according to given percentages
    Input: numpy dir -> contains file path to data folder of numpy files
           dataset_percentages -> (% train, % test) such that % train + % test = 100
           OR
           dataset_percentages -> (% train, % val, % test) such that % train + % val + % test = 100
    Output: new folders for training and testing or training/validation/testing
    '''
    
    # If just train and test
    if len(dataset_percentages) == 2:
        # Making Main data folder
        new_dir = os.path.join(os.getcwd(), output_dir)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        elif delete_previous == True:
            shutil.rmtree(new_dir)
            os.mkdir(new_dir)
        
        # Making train subfolder
        train_dir = os.path.join(new_dir, 'train')
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
            train_dir = os.path.join(train_dir, 'input')
            os.mkdir(train_dir)
        
        # Making test subfolder
        test_dir = os.path.join(new_dir, 'test')
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
            test_dir = os.path.join(test_dir, 'input')
            os.mkdir(test_dir)

        file_list = os.listdir(input_dir)
        total_num_imgs = len(file_list)
        train_percent = dataset_percentages[0]
        test_percent = dataset_percentages[1]
        valid_inputs = (train_percent >= test_percent and train_percent <= 100 and
                        test_percent <= 100 and train_percent > 0 and test_percent > 0 and
                        train_percent + test_percent == 100)
        if valid_inputs:
            num_train = int(round(total_num_imgs * train_percent//100))
        else:
            num_train = int(round(total_num_imgs * 0.9))
            print('ERROR: Please input valid percentages for dataset division')
            print('In place of valid input the ratio 90% train, 10% test was used')
        
        index = 0
        random.shuffle(file_list)
        for file in file_list:
            filename = os.fsdecode(file)
            filepath = os.path.join(input_dir, filename)
            # Loads File
            if filepath.endswith('.npy'):
                array = np.load(filepath)
                if len(array.shape) == 3:
                    array = rgb2gray(array.astype(np.float32))
                array = exposure.rescale_intensity(array, in_range='image', 
                                                   out_range=(0.0,scale))
            else:
                array = imageio.imread(filepath)
                if len(array.shape) == 3:
                    array = rgb2gray(array.astype(np.float32))
                array = exposure.rescale_intensity(array, in_range='image', 
                                                   out_range=(0.0,scale))
            if index < num_train:
                new_filepath = os.path.join(train_dir, filename)
            else:
                new_filepath = os.path.join(test_dir, filename)
            # Saves File
            new_filepath = Path(new_filepath)
            new_filepath = new_filepath.with_suffix('')
            new_filepath = new_filepath.with_suffix(file_format)
            
            if file_format == '.npy':
                np.save(new_filepath, array, allow_pickle=True, fix_imports=True)
            elif file_format == '.tif' or file_format == '.tiff':
                imageio.imwrite(new_filepath, array, file_format)
            else:
                array = exposure.rescale_intensity(array, in_range='image', 
                                                   out_range=(0.0,255.0)).astype(np.uint8)
                imageio.imwrite(new_filepath, array, file_format)
            index += 1
        return train_dir, test_dir
    # If train, val, and test
    elif len(dataset_percentages) == 3:
        # Making Main data folder
        new_dir = os.path.join(os.getcwd(), output_dir)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        elif delete_previous == True:
            shutil.rmtree(new_dir)
            os.mkdir(new_dir)
            
        # Making train subfolder
        train_dir = os.path.join(new_dir, 'train')
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
            train_dir = os.path.join(train_dir, 'input')
            os.mkdir(train_dir)
        
        # Making val subfolder
        val_dir = os.path.join(new_dir, 'val')
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)
            val_dir = os.path.join(val_dir, 'input')
            os.mkdir(val_dir)
        
        # Making test subfolder
        test_dir = os.path.join(new_dir, 'test')
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
            test_dir = os.path.join(test_dir, 'input')
            os.mkdir(test_dir)
            
        file_list = os.listdir(input_dir)
        total_num_imgs = len(file_list)
        train_percent = dataset_percentages[0]
        val_percent = dataset_percentages[1]
        test_percent = dataset_percentages[2]
        valid_inputs = (train_percent >= test_percent and train_percent >= val_percent 
                        and train_percent <= 100 and val_percent <= 100 and test_percent <= 100
                        and train_percent > 0 and val_percent > 0 and test_percent > 0 and
                        train_percent + val_percent + test_percent == 100)
        if valid_inputs:
            num_train = int(round(total_num_imgs * train_percent//100))
            num_val = int(round(total_num_imgs * val_percent//100))
        else:
            num_train = int(round(total_num_imgs * 0.9))
            num_val = int(round((total_num_imgs - num_train)/2))
            print('ERROR: Please input valid percentages for dataset division')
            print('In place of a valid input the ratio 90% train, 5% val, 5% test was used')
        
        index = 0
        random.shuffle(file_list)
        for file in file_list:
            filename = os.fsdecode(file)
            filepath = os.path.join(input_dir, filename)
            # Loads File
            if filepath.endswith('.npy'):
                array = np.load(filepath)
                if len(array.shape) == 3:
                    array = rgb2gray(array.astype(np.float32))
                array = exposure.rescale_intensity(array, in_range='image', 
                                                   out_range=(0.0,scale))
            else:
                array = imageio.imread(filepath)
                if len(array.shape) == 3:
                    array = rgb2gray(array.astype(np.float32))
                array = exposure.rescale_intensity(array, in_range='image', 
                                                   out_range=(0.0,scale))
            if index < num_train:
                new_filepath = os.path.join(train_dir, filename)
            elif index <= num_train + num_val:
                new_filepath = os.path.join(val_dir, filename)
            else:
                new_filepath = os.path.join(test_dir, filename)
            # Saves File
            new_filepath = Path(new_filepath)
            new_filepath = new_filepath.with_suffix('')
            new_filepath = new_filepath.with_suffix(file_format)
            
            if file_format == '.npy':
                np.save(new_filepath, array, allow_pickle=True, fix_imports=True)
            elif file_format == '.tif' or file_format == '.tiff':
                imageio.imwrite(new_filepath, array, file_format)
            else:
                array = exposure.rescale_intensity(array, in_range='image', 
                                                   out_range=(0.0,255.0)).astype(np.uint8)
                imageio.imwrite(new_filepath, array, file_format)
            index += 1
        return train_dir, val_dir, test_dir
    else:
        print('ERROR: Please divide into train/test or train/val/test')
        return None