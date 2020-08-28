# Customary Imports:
import tensorflow as tf
assert '2.' in tf.__version__  # make sure you're using tf 2.0
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import skimage
from skimage import exposure, transform
import cv2 as cv
import os
import datetime
import scipy
import random
import shutil
import PIL
import imageio
from pathlib import Path
from PIL import Image
import utils.patchwork_alg

##################################################################################################################################
'''
PATCHWORK ALGORITHM FUNCTIONS:
'''
def expand_image(down_image, downsampling_ratio = (1, 5), downsampling_axis = 'both', output_shape = None):
    '''
    This function expands a given image according to inputted ratio by resizing with fill_value = 0.
    By performing this operation the function seeks to place zero value pixels where downsampling
    occured within the image during image acquisition.
    '''
    if len(down_image.shape) != 0:
        if downsampling_ratio[0]==0:
            downsampling_ratio[0]=1
        if downsampling_ratio[1]==0:
            downsampling_ratio[1]=1
        i_shape = down_image.shape[0]
        j_shape = down_image.shape[1]
        if output_shape == None:
            if downsampling_axis == 'x':
                i_shape_desired = i_shape
                j_shape_desired = int(j_shape * downsampling_ratio[1])
            elif downsampling_axis == 'y':
                i_shape_desired = int(i_shape * downsampling_ratio[0])
                j_shape_desired = j_shape
            elif downsampling_axis == 'both':
                i_shape_desired = int(i_shape * downsampling_ratio[0])
                j_shape_desired = int(j_shape * downsampling_ratio[1])
            else:
                print('ERROR: Please input x or y as downsampling axis')
        else:
            if output_shape[0] >= i_shape and output_shape[1] >= j_shape:
                i_shape_desired = output_shape[0]
                j_shape_desired = output_shape[1]
            else: 
                i_shape_desired = i_shape
                j_shape_desired = j_shape
        full_image = np.zeros((i_shape_desired, j_shape_desired), dtype = np.float32)
        #print(full_image.shape)
        #print(full_image.shape)
        if downsampling_axis == 'x':
            downsampling_ratio = downsampling_ratio[1]
            for j in range(full_image.shape[1]):
                if j%downsampling_ratio==0:
                    full_image[:, j] = down_image[:, j//downsampling_ratio]
        elif downsampling_axis == 'y':
            downsampling_ratio = downsampling_ratio[0]
            for i in range(full_image.shape[0]):
                if i%downsampling_ratio==0:
                    full_image[i, :] = down_image[i//downsampling_ratio, :]
        elif downsampling_axis == 'both':
            downsampling_ratio_j = downsampling_ratio[1]
            downsampling_ratio_i = downsampling_ratio[0]
            for i in range(full_image.shape[0]):
                for j in range(full_image.shape[1]):
                    if i%downsampling_ratio_i==0 and j%downsampling_ratio_j==0:
                        full_image[i, j] = down_image[i//downsampling_ratio_i, 
                                                      j//downsampling_ratio_j]
        else:
            print('ERROR: Please input x or y as downsampling axis')
        full_image = (full_image - full_image.min()) / (full_image.max() - full_image.min())
    else:
        full_image = down_image
    return full_image

def expand_image_with_interp(down_image, downsampling_ratio = [2, 5], downsampling_axis = 'both', output_shape = None):
    '''
    This function expands a given image according to inputted ratio by resizing with bicubic interpolation.
    By performing this operation the function seeks to create a blurred approximation of the true fully-sampled image.
    '''
    if len(down_image.shape) != 0:
        if downsampling_ratio[0]==0:
            downsampling_ratio[0]=1
        if downsampling_ratio[1]==0:
            downsampling_ratio[1]=1
        i_shape = down_image.shape[0]
        j_shape = down_image.shape[1]
        if output_shape == None:
            if downsampling_axis == 'x':
                i_shape_desired = i_shape
                j_shape_desired = int(j_shape * downsampling_ratio[1])
            elif downsampling_axis == 'y':
                i_shape_desired = int(i_shape * downsampling_ratio[0])
                j_shape_desired = j_shape
            elif downsampling_axis == 'both':
                i_shape_desired = int(i_shape * downsampling_ratio[0])
                j_shape_desired = int(j_shape * downsampling_ratio[1])
            else:
                print('ERROR: Please input x or y as downsampling axis')
        else:
            if output_shape[0] >= i_shape and output_shape[1] >= j_shape:
                i_shape_desired = output_shape[0]
                j_shape_desired = output_shape[1]
            else: 
                i_shape_desired = i_shape
                j_shape_desired = j_shape
        # Bicubic Interpolation
        full_image = skimage.transform.resize(down_image, output_shape=[i_shape_desired, j_shape_desired], 
                                              order=3, mode='reflect', cval=0, clip=True, preserve_range=True, 
                                              anti_aliasing=True, anti_aliasing_sigma=None)
        #print(full_image.shape)
        full_image = (full_image - full_image.min()) / (full_image.max() - full_image.min())
    else:
        full_image = down_image
    return full_image

def fix_boundaries(orig_img, patch_img, model=None, i_count=0, j_count=0, 
                   pad_image_shape=(128,128), model_input_shape = (128,128), bound_buff = 4):
    '''
    This function augments the patchwork algorithm and makes sure the seams between the patches do not have any undue edge
    distortion.
    '''
    if len(orig_img.shape) != 0 and len(patch_img.shape) != 0 and model != None:
        img = patch_img
        if i_count == 1:
            if j_count > 1:
                batch = np.stack(np.split(orig_img[:, model_input_shape[1]//2 : pad_image_shape[1]-model_input_shape[1]//2], 
                                          orig_img.shape[1]/model_input_shape[1], axis=1), axis=0)
                pred = model.predict(batch[..., None], batch_size=batch.shape[0])
                pred = pred[...,0]
                strip_count = 0
                for j in range(model_input_shape[1]//2,pad_image_shape[1]-model_input_shape[1]//2,model_input_shape[1]):
                    #plt.imshow(pred, cmap='gray')
                    #plt.show()
                    mid_patch = j+model_input_shape[1]//2
                    img[:,mid_patch-bound_buff:mid_patch+bound_buff] = pred[strip_count, :, pred.shape[2]//2-bound_buff:pred.shape[2]//2+bound_buff]
                    strip_count+=1
        else:
            for i in range(0, pad_image_shape[0], model_input_shape[0]):
                strip = orig_img[i:i+model_input_shape[0], model_input_shape[1]//2 : pad_image_shape[1]-model_input_shape[1]//2]
                batch = np.stack(np.split(strip, strip.shape[1]/model_input_shape[1], axis=1), axis=0)
                pred = model.predict(batch[...,None], batch_size=batch.shape[0])
                pred = pred[...,0]
                strip_count = 0
                for j in range(model_input_shape[1]//2, pad_image_shape[1]-model_input_shape[1]//2, model_input_shape[1]):
                    #plt.imshow(pred, cmap='gray')
                    #plt.show()
                    mid_patch = j+model_input_shape[1]//2
                    #plt.imshow(img[i:i+model_input_shape[1],mid_patch-bound_buff:mid_patch+bound_buff], cmap='gray')
                    #plt.show()
                    #'''
                    # TO NOT SHOW PATCH PATTERN
                    img[i:i+model_input_shape[0],
                        mid_patch-bound_buff:mid_patch+bound_buff] = pred[strip_count, :, model_input_shape[1]//2-bound_buff:model_input_shape[1]//2+bound_buff]
                    strip_count+=1
                    #'''
                    '''
                    # TO SHOW PATCH PATTERN
                    img[i:i+model_input_shape[0],
                        mid_patch-bound_buff:mid_patch+bound_buff] = np.ones(pred[:, pred.shape[2]//2-bound_buff:
                                                                                  pred.shape[2]//2+bound_buff].shape)
                    '''
                    #plt.imshow(img[i:i+model_input_shape[0],mid_patch-bound_buff:mid_patch+bound_buff], cmap='gray')
                    #plt.show()
            for j in range(0,pad_image_shape[1],model_input_shape[1]):
                strip = orig_img[model_input_shape[0]//2:pad_image_shape[0]-model_input_shape[0]//2, j:j+model_input_shape[1]]
                batch = np.stack(np.split(strip, strip.shape[0]/model_input_shape[0], axis=0), axis=0)
                pred = model.predict(batch[..., None], batch_size=batch.shape[0])
                pred = pred[...,0]
                strip_count = 0
                for i in range(model_input_shape[0]//2,pad_image_shape[0]-model_input_shape[0]//2,model_input_shape[0]):
                    #plt.imshow(pred, cmap='gray')
                    #plt.show()
                    mid_patch = i+model_input_shape[0]//2
                    #'''
                    # TO NOT SHOW PATCH PATTERN
                    img[mid_patch-bound_buff:mid_patch+bound_buff, 
                        j:j+model_input_shape[1]] = pred[strip_count, model_input_shape[0]//2-bound_buff:model_input_shape[0]//2+bound_buff, :]
                    strip_count+=1
                    #'''
                    '''
                    # TO SHOW PATCH PATTERN
                    img[mid_patch-bound_buff:mid_patch+bound_buff, 
                        j:j+model_input_shape[1]] = np.zeros(pred[pred.shape[1]//2-bound_buff:
                                                                  pred.shape[1]//2+bound_buff,:].shape)
                    '''
            # Cover Overlap
            pad = bound_buff//4+1
            bound_buff += pad
            for i in range(model_input_shape[0]//2,pad_image_shape[0]-model_input_shape[0]//2,model_input_shape[0]):
                strip = orig_img[i:i+model_input_shape[0], model_input_shape[1]//2:pad_image_shape[1]-model_input_shape[1]//2]
                batch = np.stack(np.split(strip, strip.shape[1]/model_input_shape[1], axis=1), axis=0)
                pred = model.predict(batch[..., None], batch_size=batch.shape[0])
                pred = pred[...,0]
                strip_count = 0
                for j in range(model_input_shape[1]//2,pad_image_shape[1]-model_input_shape[1]//2,model_input_shape[1]):
                    #plt.imshow(pred, cmap='gray')
                    #plt.show()
                    mid_patch_i = i+model_input_shape[0]//2
                    mid_patch_j = j+model_input_shape[1]//2
                    #plt.imshow(img[i:i+model_input_shape[1],mid_patch-bound_buff:mid_patch+bound_buff], cmap='gray')
                    #plt.show()
                    #'''
                    # TO NOT SHOW PATCH PATTERN
                    img[mid_patch_i-bound_buff:mid_patch_i+bound_buff,
                        mid_patch_j-bound_buff:mid_patch_j+bound_buff] = pred[strip_count, 
                                                                              pred.shape[1]//2-bound_buff:pred.shape[1]//2+bound_buff, 
                                                                              pred.shape[2]//2-bound_buff:pred.shape[2]//2+bound_buff]
                    strip_count+=1
                    #'''
                    '''
                    # TO SHOW PATCH PATTERN
                    img[mid_patch_i-bound_buff:mid_patch_i+bound_buff,
                        mid_patch_j-bound_buff:mid_patch_j+bound_buff] = 0.5*np.ones(pred[pred.shape[1]//2-bound_buff:pred.shape[1]//2+bound_buff, 
                                                                                          pred.shape[2]//2-bound_buff:pred.shape[2]//2+bound_buff].shape)
                    '''
                    #plt.imshow(img[i:i+model_input_shape[0],mid_patch-bound_buff:mid_patch+bound_buff], cmap='gray')
                    #plt.show()
    else:
        img = orig_img
    return img

def remove_peak(image, num_std = 4):
    MAX_STD = 10
    image = (image - image.min()) / (image.max() - image.min())
    orig_image = image
    if num_std > MAX_STD:
        num_std = MAX_STD
    mean = np.mean(image)
    std = np.std(image)
    #print(mean)
    #print(std)
    step = -0.001
    for std_lev in np.arange(MAX_STD, num_std, step):
        threshold = mean + std_lev*std
        if threshold > 1:
            threshold = 1
        decay_factor = 0.1
        if len(image[image > threshold].shape) > 0:
            image[image > threshold] = np.mean(image[image > threshold])*(1 - decay_factor)
    image = (image - image.min()) / (image.max() - image.min())
    '''
    # SHOW CLEANED IMAGE:
    figsize = (15,15)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(image, cmap = 'gray')
    plt.title('Cleaned Latent Image')
    '''
    return image

def remove_padding(img):
    # Removing Image Border (if it exists)
    pad_img = img
    if len(img.shape)>2:
        pad_img = img[...,0]
    xmin=0
    xmax=pad_img.shape[0]-1
    ymin=0
    ymax=pad_img.shape[1]-1
    for xmin in range(pad_img.shape[0]):
        if np.sum(pad_img[xmin,...]) > 0.01:
            break
    for xmax in range(pad_img.shape[0]-1, 0, -1):
        if np.sum(pad_img[xmax,...]) > 0.01:
            break
    for ymin in range(pad_img.shape[1]):
        if np.sum(pad_img[:,ymin,...]) > 0.01:
            break
    for ymax in range(pad_img.shape[1]-1, 0, -1):
        if np.sum(pad_img[:,ymax,...]) > 0.01:
            break
    no_pad = img[xmin:xmax,ymin:ymax+1,...]
    return no_pad
##################################################################################################################################
'''
PATCHWORK ALGORITHM:
'''
def apply_model_patchwork(model, down_image, downsampling_ratio = (1,5), downsampling_axis = 'both', shape_for_model = (128,128), 
                          buffer = 10, output_shape = None, interp = False, remove_pad = False, gauss_blur_std=None, single_pass = False):
    '''
    This function expands the image and then performs the model patchwork algorithm. Patches
    of the image are extracted and processed by the given CNN model.
    '''
    # This Function Expands the Image and Then Performs Model Patchwork Algorithm
    STARTING_POINT = (0,0)
    i_shape = down_image.shape[0]
    j_shape = down_image.shape[1]
    if interp:
        full_image = expand_image_with_interp(down_image, downsampling_ratio = downsampling_ratio, 
                                              downsampling_axis = downsampling_axis, output_shape = output_shape)
    else:
        full_image = expand_image(down_image, downsampling_ratio = downsampling_ratio, 
                                  downsampling_axis = downsampling_axis, output_shape = output_shape)
    full_i_shape = full_image.shape[0]
    full_j_shape = full_image.shape[1]
    default_pad = np.max(shape_for_model)//2
    #default_pad = 0
    if full_i_shape%shape_for_model[0] != 0:
        i_left = full_i_shape%shape_for_model[0]
        i_pad = (shape_for_model[0] - i_left)//2
        rest_i = (shape_for_model[0] - i_left)%2
    else:
        i_left = 0
        i_pad = default_pad
        rest_i = 0
    if full_j_shape%shape_for_model[1] != 0:
        j_left = full_j_shape%shape_for_model[1]
        j_pad = (shape_for_model[1] - j_left)//2
        rest_j = (shape_for_model[1] - j_left)%2
    else:
        j_left = 0
        j_pad = default_pad
        rest_j = 0
    
    #print('i_left = '+str(i_left))
    #print('j_left = '+str(j_left))
    #print('i_pad = '+str(i_pad))
    #print('j_pad = '+str(j_pad))
    #print('rest_i = '+str(rest_i))
    #print('rest_j = '+str(rest_j))
    pad_image = np.pad(full_image, [(i_pad, ), (j_pad, )], mode='constant')
    full_pad_image = np.pad(pad_image, [(0, rest_i), (0, rest_j)], mode='constant')
    
    if gauss_blur_std is not None:
        full_pad_image = scipy.ndimage.gaussian_filter(full_pad_image, sigma=gauss_blur_std, order=0, 
                                                       output=None, mode='reflect', cval=0.0, truncate=6.0)
    '''
    figsize = (15,15)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(full_pad_image, cmap = 'gray')
    plt.show()
    '''
    
    orig_image_j_start = j_pad #+ rest_j
    orig_image_i_start = i_pad #+ rest_i
    orig_image_j_end = orig_image_j_start + full_j_shape
    orig_image_i_end = orig_image_i_start + full_i_shape
    #print(orig_image_i_start)
    #print(orig_image_i_end)
    #print(orig_image_j_start)
    #print(orig_image_j_end)
    full_patch_image = np.ones(full_pad_image.shape)
    for i in range(0, full_pad_image.shape[0], shape_for_model[0]):
        '''
        batch = np.zeros((full_pad_image.shape[1]//shape_for_model[1], 
                          shape_for_model[0], shape_for_model[1]))
        '''
        strip = full_pad_image[i:i+shape_for_model[0], :]
        batch = np.stack(np.split(strip, full_pad_image.shape[1]/shape_for_model[1], axis=1), axis=0)
        batch = batch[..., None]
        pred = model.predict(batch, batch_size = batch.shape[0])
        full_patch_image[i:i+shape_for_model[0], :] = np.concatenate(pred[...,0], axis=1)
    
    #'''
    # To Fix Patch Boundaries With Second and Third Passes
    i_count = full_pad_image.shape[0]/shape_for_model[0]
    j_count = full_pad_image.shape[1]/shape_for_model[1]
    if (i_count > 1 or j_count > 1) and not single_pass:
        full_patch_image = fix_boundaries(full_pad_image, full_patch_image, model, i_count, j_count, pad_image_shape = full_pad_image.shape,
                                          model_input_shape = shape_for_model, bound_buff = buffer)
    #'''
    #start_i_temp = orig_image_i_start + i_count*shape_for_model[0]
    #end_i_temp = orig_image_i_start + (i_count+1)*shape_for_model[0]
    #start_j_temp = orig_image_j_start + j_count*shape_for_model[1]
    #end_j_temp = orig_image_j_start + (j_count+1)*shape_for_model[1]
    #print('start_i_temp = '+str(start_i_temp))
    #print('end_i_temp = '+str(end_i_temp))
    #print('start_j_temp = '+str(start_j_temp))
    #print('end_j_temp = '+str(end_j_temp))
    full_recon_image = full_patch_image[orig_image_i_start:orig_image_i_end,orig_image_j_start:orig_image_j_end]
    full_recon_image = (full_recon_image - full_recon_image.min()) / (full_recon_image.max() - full_recon_image.min())
    if remove_pad:
        full_recon_image = remove_padding(full_recon_image)
        full_recon_image = (full_recon_image - full_recon_image.min()) / (full_recon_image.max() - full_recon_image.min())
    return full_recon_image