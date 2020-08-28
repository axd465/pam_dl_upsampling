# Customary Imports:
import tensorflow as tf
assert '2.' in tf.__version__  # make sure you're using tf 2.0
import numpy as np
import matplotlib.pyplot as plt
import math
import string
import pandas as pd
import sklearn
import skimage
import cv2 as cv
import os
import datetime
import scipy
from skimage import exposure, transform
import random
import re
import time
import shutil
import imageio
import copy
from pathlib import Path
from tensorflow.keras import backend as K
import PIL
from PIL import Image
from tensorflow.keras import Model

# Added Import Statements:
from utils.patchwork_alg import *
from utils.model_utils import PSNR
from utils.model_utils import SSIM
from utils.model_utils import MS_SSIM
from utils.model_utils import KLDivergence
import utils.test_statistics
##################################################################################################################################
'''
COMPUTING TEST STATISTICS FOR INTERPOLATION:
'''
def obtain_test_stats(model, input_dir, downsampling_ratio = (1,5), shape_for_model = (128,128), 
                      buffer = 20, contrast_enhance = (0.05, 99.95), interp=False, gauss_blur_std=None,
                      output_dir = None, file_format = '.tif', remove_pad = False, latent_img_input = False,
                      i_ratio = 2, j_ratio = 1):
    '''
    This function loops through a directory and computes various test statistics (comparing DL and Bicubic
    Interpolation) for each image in the directory. These test statistics are then exported as a dictionary.
    '''
    file_list = [file for file in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, file))]
    metrics = {'FILENAME':[], 'PSNR':[], 'SSIM':[], 'MS-SSIM':[], 'MEAN ABSOLUTE ERROR':[], 'MEAN SQUARED ERROR':[]}
    '''
    metrics = {'FILENAME':[], 'PSNR':[], 'SSIM':[], 'MS-SSIM':[], 'MEAN ABSOLUTE ERROR':[], 'MEAN SQUARED ERROR':[], 
               'KL DIVERGENCE':[], 'MEAN SQUARED LOG ERROR':[], 'LOG COSH ERROR':[], 
               'POISSON LOSS':[]}
    '''
    stats = {'Deep Learning':copy.deepcopy(metrics), 'Bicubic Interpolation':copy.deepcopy(metrics),
             'Lanczos Interpolation':copy.deepcopy(metrics), 'Zero Fill':copy.deepcopy(metrics)}
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        else:
            shutil.rmtree(output_dir)
            os.mkdir(output_dir)
        dl_dir = os.path.join(output_dir, 'deep_learning')
        os.mkdir(dl_dir)
        bc_dir = os.path.join(output_dir, 'bicubic_interp')
        os.mkdir(bc_dir)
        lanczos_dir = os.path.join(output_dir, 'lanczos_interp')
        os.mkdir(lanczos_dir)
        zero_dir = os.path.join(output_dir, 'zero_fill')
        os.mkdir(zero_dir)
        if not latent_img_input:
            full_sample_dir = os.path.join(output_dir, 'fully_sampled')
            os.mkdir(full_sample_dir)
        input_img_dir = os.path.join(output_dir, 'input_imgs')
        os.mkdir(input_img_dir)
        latent_dir = os.path.join(output_dir, 'latent_imgs')
        os.mkdir(latent_dir)
        mask_dir = os.path.join(output_dir, 'masks')
        os.mkdir(mask_dir)
    total_times=[]
    model_times=[]
    shape_list=[]
    file_count = 1
    for file in file_list:
        # Timing Procedure (Outer Procedure Start):
        start_tot = time.time()
        # Load Image
        filename = os.fsdecode(file)
        filepath = os.path.join(input_dir, filename)
        if filepath.endswith('.npy'):
            img = np.load(filepath)
        else:
            img = imageio.imread(filepath)
            img = np.array(img, dtype=np.float32)
        # Recover Latent Image:
        if latent_img_input:
            full_samp_img = (img - img.min()) / (img.max() - img.min())
            latent_image = full_samp_img.copy()
            shape = [full_samp_img.shape[0]*downsampling_ratio[0], 
                     full_samp_img.shape[1]*downsampling_ratio[1]]
            zero_fill = np.zeros(shape)
            for i in range(0, zero_fill.shape[0]):
                for j in range(0, zero_fill.shape[1]):
                    if i%downsampling_ratio[0]==0 and j%downsampling_ratio[1]==0:
                        zero_fill[i, j] = latent_image[i//downsampling_ratio[0], j//downsampling_ratio[1]]
            mask = np.zeros(shape)
            mask[::downsampling_ratio[0], ::downsampling_ratio[1]] = 1
        else:
            full_samp_img = img
            if len(full_samp_img.shape) > 2:
                full_samp_img = full_samp_img[...,0]
            full_samp_img = (full_samp_img - full_samp_img.min()) / (full_samp_img.max() - full_samp_img.min())
            '''
            # If using color images
            if len(img.shape)>2:
                img = np.mean(img, axis = 2)
            '''
            # Resizing Images To Balance Downsampling Ratio Along Axes (one axis already downsampled):
            full_samp_img_shape = (full_samp_img.shape[0]*i_ratio, full_samp_img.shape[1]*j_ratio)
            shape_list.append(full_samp_img_shape)
            full_samp_img = skimage.transform.resize(full_samp_img, output_shape=full_samp_img_shape, order=3, 
                                                     mode='reflect', cval=0, clip=True, preserve_range=True, 
                                                     anti_aliasing=True, anti_aliasing_sigma=None)
            shape = full_samp_img.shape
            mask = np.zeros(shape, dtype=np.int32)
            mask[::downsampling_ratio[0], ::downsampling_ratio[1]] = 1
            zero_fill = full_samp_img*mask
            latent_image = full_samp_img[::downsampling_ratio[0], ::downsampling_ratio[1]]
        # Timing Procedure (Inner Procedure Start):
        start_mod = time.time()
        deep_image = apply_model_patchwork(model, down_image = latent_image, downsampling_ratio = downsampling_ratio, 
                                           downsampling_axis = 'both', shape_for_model = shape_for_model, 
                                           buffer = buffer, output_shape = shape, interp = interp, 
                                           remove_pad = remove_pad, gauss_blur_std = gauss_blur_std)
        # Timing Procedure (Inner Procedure End):
        model_times.append(time.time() - start_mod)
        #'''
        if contrast_enhance is not None:
            deep_image = skimage.img_as_float(deep_image)
            p1, p2 = np.percentile(deep_image, contrast_enhance)
            deep_image = exposure.rescale_intensity(deep_image, in_range=(p1, p2), out_range=(0.0,1.0))
        #'''

        # COMPARISON TO INTERPOLATION:
        # From https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.warp
        # and https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize
        # Bicubic Interpolation:
        bc_interp_img = skimage.transform.resize(latent_image, output_shape=shape, order=3, mode='reflect', 
                                                 cval=0, clip=True, preserve_range=True, anti_aliasing=True, anti_aliasing_sigma=None)
        # Lanzcos Interpolation:
        norm_down_img = (latent_image - latent_image.min()) / (latent_image.max() - latent_image.min()) * 255
        lanczos = cv.resize(norm_down_img.astype(np.uint8), dsize=tuple(shape[::-1]), interpolation = cv.INTER_LANCZOS4).astype(np.float32)
        lanczos = (lanczos - lanczos.min()) / (lanczos.max() - lanczos.min())
        '''
        # To Show Images
        figsize = (20,20)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(deep_image, cmap = 'gray')
        plt.title('DL Image')
        plt.show()
        '''
        if latent_img_input:
            input_img = np.stack([np.zeros(shape), zero_fill, mask], axis=-1)
        else:
            input_img = np.stack([full_samp_img, zero_fill, mask], axis=-1)
        # Save Images to Output Directory
        if output_dir is not None:
            test_statistics.save_img(deep_image, file, dl_dir, downsampling_ratio, file_format)
            test_statistics.save_img(bc_interp_img, file, bc_dir, downsampling_ratio, file_format)
            test_statistics.save_img(lanczos, file, lanczos_dir, downsampling_ratio, file_format)
            test_statistics.save_img(zero_fill, file, zero_dir, downsampling_ratio, file_format)
            if not latent_img_input:
                test_statistics.save_img(full_samp_img, file, full_sample_dir, downsampling_ratio, file_format)
            test_statistics.save_img(input_img, file, input_img_dir, downsampling_ratio, file_format)
            test_statistics.save_img(latent_image, file, latent_dir, downsampling_ratio, file_format)
            test_statistics.save_img(mask, file, mask_dir, downsampling_ratio, file_format)

        # Quantitative Measurements
        if not latent_img_input:
            deep_image = deep_image[..., None]
            deep_image = tf.image.convert_image_dtype(deep_image[None, ...], tf.float32)
            bc_interp_img = bc_interp_img[..., None]
            bc_interp_img = tf.image.convert_image_dtype(bc_interp_img[None, ...], tf.float32)
            lanczos = lanczos[..., None]
            lanczos = tf.image.convert_image_dtype(lanczos[None, ...], tf.float32)
            zero_fill = zero_fill[..., None]
            zero_fill = tf.image.convert_image_dtype(zero_fill[None, ...], tf.float32)
            full_samp_img = full_samp_img[..., None]
            full_samp_img = tf.image.convert_image_dtype(full_samp_img[None, ...], tf.float32)
            '''
            ---------------------------------------
            ########## LIST OF STATISTICS: ##########
            ---------------------------------------
            '''
            # FILENAME:
            stats['Deep Learning']['FILENAME'].extend([file])
            stats['Bicubic Interpolation']['FILENAME'].extend([file])
            stats['Lanczos Interpolation']['FILENAME'].extend([file])
            stats['Zero Fill']['FILENAME'].extend([file])
            # PSNR:
            stats['Deep Learning']['PSNR'].extend(PSNR(full_samp_img, deep_image).numpy())
            stats['Bicubic Interpolation']['PSNR'].extend(PSNR(full_samp_img, bc_interp_img).numpy())
            stats['Lanczos Interpolation']['PSNR'].extend(PSNR(full_samp_img, lanczos).numpy())
            stats['Zero Fill']['PSNR'].extend(PSNR(full_samp_img, zero_fill).numpy())
            # SSIM:
            stats['Deep Learning']['SSIM'].extend(SSIM(full_samp_img, deep_image).numpy())
            stats['Bicubic Interpolation']['SSIM'].extend(SSIM(full_samp_img, bc_interp_img).numpy())
            stats['Lanczos Interpolation']['SSIM'].extend(SSIM(full_samp_img, lanczos).numpy())
            stats['Zero Fill']['SSIM'].extend(SSIM(full_samp_img, zero_fill).numpy())
            # MS-SSIM:
            stats['Deep Learning']['MS-SSIM'].extend(MS_SSIM(full_samp_img, deep_image).numpy())
            stats['Bicubic Interpolation']['MS-SSIM'].extend(MS_SSIM(full_samp_img, bc_interp_img).numpy())
            stats['Lanczos Interpolation']['MS-SSIM'].extend(MS_SSIM(full_samp_img, lanczos).numpy())
            stats['Zero Fill']['MS-SSIM'].extend(MS_SSIM(full_samp_img, zero_fill).numpy())
            # MAE:
            MAE = tf.keras.losses.MeanAbsoluteError()
            stats['Deep Learning']['MEAN ABSOLUTE ERROR'].extend([MAE(full_samp_img, deep_image).numpy()])
            stats['Bicubic Interpolation']['MEAN ABSOLUTE ERROR'].extend([MAE(full_samp_img, bc_interp_img).numpy()])
            stats['Lanczos Interpolation']['MEAN ABSOLUTE ERROR'].extend([MAE(full_samp_img, lanczos).numpy()])
            stats['Zero Fill']['MEAN ABSOLUTE ERROR'].extend([MAE(full_samp_img, zero_fill).numpy()])
            # MSE:
            MSE = tf.keras.losses.MeanSquaredError()
            stats['Deep Learning']['MEAN SQUARED ERROR'].extend([MSE(full_samp_img, deep_image).numpy()])
            stats['Bicubic Interpolation']['MEAN SQUARED ERROR'].extend([MSE(full_samp_img, bc_interp_img).numpy()])
            stats['Lanczos Interpolation']['MEAN SQUARED ERROR'].extend([MSE(full_samp_img, lanczos).numpy()])
            stats['Zero Fill']['MEAN SQUARED ERROR'].extend([MSE(full_samp_img, zero_fill).numpy()])
            '''
            ---------------------------------------
            ########## FURTHER METRICS: ##########
            ---------------------------------------
            # KL DIVERGENCE:
            stats['Deep Learning']['KL DIVERGENCE'].extend([KLDivergence(full_samp_img, deep_image).numpy()])
            stats['Bicubic Interpolation']['KL DIVERGENCE'].extend([KLDivergence(full_samp_img, bc_interp_img).numpy()])
            stats['Lanczos Interpolation']['KL DIVERGENCE'].extend([KLDivergence(full_samp_img, lanczos).numpy()])
            stats['Zero Fill']['KL DIVERGENCE'].extend([KLDivergence(full_samp_img, zero_fill).numpy()])
            # MEAN SQUARED LOGARITHMIC ERROR:
            MSLE = tf.keras.losses.MeanSquaredLogarithmicError()
            stats['Deep Learning']['MEAN SQUARED LOG ERROR'].extend([MSLE(full_samp_img, deep_image).numpy()])
            stats['Bicubic Interpolation']['MEAN SQUARED LOG ERROR'].extend([MSLE(full_samp_img, bc_interp_img).numpy()])
            stats['Lanczos Interpolation']['MEAN SQUARED LOG ERROR'].extend([MSLE(full_samp_img, lanczos).numpy()])
            stats['Zero Fill']['MEAN SQUARED LOG ERROR'].extend([MSLE(full_samp_img, zero_fill).numpy()])
            # LOG COSH ERROR:
            LCOSH = tf.keras.losses.LogCosh()
            stats['Deep Learning']['LOG COSH ERROR'].extend([LCOSH(full_samp_img, deep_image).numpy()])
            stats['Bicubic Interpolation']['LOG COSH ERROR'].extend([LCOSH(full_samp_img, bc_interp_img).numpy()])
            stats['Lanczos Interpolation']['LOG COSH ERROR'].extend([LCOSH(full_samp_img, lanczos).numpy()])
            stats['Zero Fill']['LOG COSH ERROR'].extend([LCOSH(full_samp_img, zero_fill).numpy()])
            # POISSON LOSS:
            PL = tf.keras.losses.Poisson()
            stats['Deep Learning']['POISSON LOSS'].extend([PL(full_samp_img, deep_image).numpy()])
            stats['Bicubic Interpolation']['POISSON LOSS'].extend([PL(full_samp_img, bc_interp_img).numpy()])
            stats['Lanczos Interpolation']['POISSON LOSS'].extend([PL(full_samp_img, lanczos).numpy()])
            stats['Zero Fill']['POISSON LOSS'].extend([PL(full_samp_img, zero_fill).numpy()])
            '''
        print(f'Done with file {file_count} out of {len(file_list)}')
        file_count += 1
        # Timing Procedure (Outer Procedure End):
        total_times.append(time.time() - start_tot)
    return stats, model_times, total_times, np.array(shape_list)

def save_img(img, filename, output_dir, down_ratio, file_format='.tif'):
        file = filename
        if file[-5].isdigit() and '-' in file[-9:]:
                    file_ext = file[-4:]
                    regex = re.compile(r'_[^_]+$')
                    file = re.sub(regex, f"_{down_ratio[0]}-{down_ratio[1]}", file) + file_ext
        filepath = os.path.join(output_dir, file)
        # Save Image
        if file_format == '.npy':
            new_filepath = Path(filepath)
            new_filepath = new_filepath.with_suffix('')
            new_filepath = new_filepath.with_suffix(file_format)
            np.save(new_filepath, img, allow_pickle=True, fix_imports=True)
        elif file_format == '.tif':
            new_filepath = Path(filepath)
            new_filepath = new_filepath.with_suffix('')
            new_filepath = new_filepath.with_suffix(file_format)
            imageio.imwrite(new_filepath, img)
        else:
            temp = ((img - img.min()) / (img.max() - img.min()) * 255).astype('uint8')
            new_filepath = Path(filepath)
            new_filepath = new_filepath.with_suffix('')
            new_filepath = new_filepath.with_suffix(file_format)
            imageio.imwrite(new_filepath, temp)
            
def display_stats(stats, model_times, total_times, shape_list, directory=None):
    # Displaying Avg Img Size and Runtimes
    print((f'Average Image Size: ({int(np.round(np.mean(shape_list[:,0]),0))} \u00B1 {np.round(np.std(shape_list[:,0]),1)}, ') +
          (f'{int(np.round(np.mean(shape_list[:,1]),0))} \u00B1 {np.round(np.std(shape_list[:,1]),1)})'))
    print(f'Average Model Patchwork Runtime: {np.round(np.mean(model_times),3)} \u00B1 {np.round(np.std(model_times),4)} seconds')
    print(f'Average Total Stats Loop Runtime (includes I/O): {np.round(np.mean(total_times),3)} \u00B1 {np.round(np.std(total_times),4)} seconds')

    # Display Stats Data:
    metric = 'SSIM'
    print_stats(metric, stats)
    metric = 'MS-SSIM'
    print_stats(metric, stats)
    metric = 'PSNR'
    print_stats(metric, stats)
    metric = 'MEAN ABSOLUTE ERROR'
    print_stats(metric, stats)
    metric = 'MEAN SQUARED ERROR'
    print_stats(metric, stats)
    
    if directory is not None:
        file = os.path.join(directory, "stats_data_output.txt")
        with open(file, 'w') as text_file:
            # Displaying Avg Img Size and Runtimes
            print((f'Average Image Size: ({int(np.round(np.mean(shape_list[:,0]),0))} \u00B1 {np.round(np.std(shape_list[:,0]),1)}, ') +
                  (f'{int(np.round(np.mean(shape_list[:,1]),0))} \u00B1 {np.round(np.std(shape_list[:,1]),1)})'), file=text_file)
            print(f'Average Model Patchwork Runtime: {np.round(np.mean(model_times),3)} \u00B1 {np.round(np.std(model_times),4)} seconds', file=text_file)
            print(f'Average Total Stats Loop Runtime (includes I/O): {np.round(np.mean(total_times),3)} \u00B1 {np.round(np.std(total_times),4)} seconds', file=text_file)

            # Display Stats Data:
            metric = 'SSIM'
            print_stats(metric, stats, text_file)
            metric = 'MS-SSIM'
            print_stats(metric, stats, text_file)
            metric = 'PSNR'
            print_stats(metric, stats, text_file)
            metric = 'MEAN ABSOLUTE ERROR'
            print_stats(metric, stats, text_file)
            metric = 'MEAN SQUARED ERROR'
            print_stats(metric, stats, text_file)
            
def print_stats(metric, stats, text_file=None):
    if text_file is not None:
        print(f'\n--------------{metric}--------------', file=text_file)
        mean = np.mean(np.array(stats['Deep Learning'][metric]))
        stdev = np.std(np.array(stats['Deep Learning'][metric]))
        print(f'Deep Learning (mean): {mean}', file=text_file)
        print(f'Deep Learning (sd): {stdev}', file=text_file)
        mean = np.mean(np.array(stats['Bicubic Interpolation'][metric]))
        stdev = np.std(np.array(stats['Bicubic Interpolation'][metric]))
        print(f'Bicubic Interpolation (mean): {mean}', file=text_file)
        print(f'Bicubic Interpolation (sd): {stdev}', file=text_file)
        mean = np.mean(np.array(stats['Lanczos Interpolation'][metric]))
        stdev = np.std(np.array(stats['Lanczos Interpolation'][metric]))
        print(f'Lanczos Interpolation (mean): {mean}', file=text_file)
        print(f'Lanczos Interpolation (sd): {stdev}', file=text_file)
        mean = np.mean(np.array(stats['Zero Fill'][metric]))
        stdev = np.std(np.array(stats['Zero Fill'][metric]))
        print(f'Zero Fill (mean): {mean}', file=text_file)
        print(f'Zero Fill (sd): {stdev}', file=text_file)
    else:
        print(f'\n--------------{metric}--------------')
        mean = np.mean(np.array(stats['Deep Learning'][metric]))
        stdev = np.std(np.array(stats['Deep Learning'][metric]))
        print(f'Deep Learning (mean): {mean}')
        print(f'Deep Learning (sd): {stdev}')
        mean = np.mean(np.array(stats['Bicubic Interpolation'][metric]))
        stdev = np.std(np.array(stats['Bicubic Interpolation'][metric]))
        print(f'Bicubic Interpolation (mean): {mean}')
        print(f'Bicubic Interpolation (sd): {stdev}')
        mean = np.mean(np.array(stats['Lanczos Interpolation'][metric]))
        stdev = np.std(np.array(stats['Lanczos Interpolation'][metric]))
        print(f'Lanczos Interpolation (mean): {mean}')
        print(f'Lanczos Interpolation (sd): {stdev}')
        mean = np.mean(np.array(stats['Zero Fill'][metric]))
        stdev = np.std(np.array(stats['Zero Fill'][metric]))
        print(f'Zero Fill (mean): {mean}')
        print(f'Zero Fill (sd): {stdev}')
