# Customary Imports:
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os
import random
import shutil
import PIL
import imageio
import keras.backend as K
from pathlib import Path
from PIL import Image
from skimage import exposure
import utils.model_utils

##################################################################################################################################
'''
MODEL UTILS:
'''
##################################################################################################################################
# Custom Metrics:

def normalize(tensor):
    # Normalizes Tensor from 0-1
    return tf.math.divide_no_nan(tf.math.subtract(tensor, tf.math.reduce_min(tensor)), 
                                 tf.math.subtract(tf.math.reduce_max(tensor), tf.math.reduce_min(tensor)))
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    y_true_norm = tf.map_fn(normalize, y_true)
    y_pred_norm = tf.map_fn(normalize, y_pred)
    PSNR = tf.image.psnr(y_true_norm, y_pred_norm, max_pixel)
    return PSNR

def SSIM(y_true, y_pred):
    max_pixel = 1.0
    y_true_norm = tf.map_fn(normalize, y_true)
    y_pred_norm = tf.map_fn(normalize, y_pred)
    SSIM = tf.image.ssim(y_true_norm,y_pred_norm,max_pixel,filter_size=11,
                         filter_sigma=1.5,k1=0.01,k2=0.03)
    return SSIM

def MS_SSIM(y_true, y_pred):
    max_pixel = 1.0
    y_true_norm = tf.map_fn(normalize, y_true)
    y_pred_norm = tf.map_fn(normalize, y_pred)
    MS_SSIM = tf.image.ssim_multiscale(y_true_norm, 
                                       y_pred_norm, 
                                       max_pixel)
    return MS_SSIM

def KLDivergence(y_true, y_pred):
    return tf.losses.KLDivergence()(y_true, y_pred)

def TV(y_true, y_pred):
    img_list = [y_true[0,...], y_pred[0,...]]
    images = tf.stack(img_list)
    loss = tf.math.reduce_sum(tf.image.total_variation(images))
    return loss

def SavingMetric(y_true, y_pred, A1=40, A2=275):
    # Combines Insight from SSIM and PSNR
    ssim = SSIM(y_true, y_pred)
    psnr = PSNR(y_true, y_pred)
    # Normalize for Minimization:
    ssim_norm = 1 - ssim
    psnr_norm = (A1 - psnr)/A2
    loss = ssim_norm + psnr_norm
    return loss

# Model Loss Function:
def model_loss(B1=1.0, B2=0.01, mse = False, name='loss_func'):
    #@tf.function
    def loss_func(y_true, y_pred):
        F_true = tf.map_fn(FFT_mag, y_true)
        F_pred = tf.map_fn(FFT_mag, y_pred)
        if mse:
            pixelwise_loss = tf.keras.losses.MeanSquaredError()
        else:
            pixelwise_loss = tf.keras.losses.MeanAbsoluteError()
        if tf.executing_eagerly():
            pixel_loss = pixelwise_loss(y_true, y_pred).numpy()
            # Fourier Loss
            F_loss = pixelwise_loss(F_true, F_pred).numpy()
        else:
            pixel_loss = pixelwise_loss(y_true, y_pred)
            # Fourier Loss
            F_loss = pixelwise_loss(F_true, F_pred)
        F_loss = tf.cast(F_loss, dtype=tf.float32)
        loss = B1*pixel_loss + B2*F_loss
        return loss
    loss_func.__name__ = name
    return loss_func

# Model Loss Function (Fourier Loss):
def FFT_mag(img):
    # FFT Function to be performed for each instance in batch
    #'''
    real = tf.cast(img, tf.float32)
    imag = tf.zeros_like(tf.cast(img, tf.float32))
    out = tf.abs(tf.signal.fftshift(tf.signal.fft2d(tf.complex(real, imag)[:, :, 0])))
    #'''
    return out

# Experimental Model Loss Function:
def model_loss_experimental(B1=1.0, B2=0.0, B3=0.0, mse=False, name='loss_func'):
    #@tf.function
    def loss_func(y_true, y_pred):
        F_true = tf.map_fn(FFT_mag, y_true)
        F_pred = tf.map_fn(FFT_mag, y_pred)
        if mse:
            pixelwise_loss = tf.keras.losses.MeanSquaredError()
        else:
            pixelwise_loss = tf.keras.losses.MeanAbsoluteError()
        if tf.executing_eagerly():
            pixel_loss = pixelwise_loss(y_true, y_pred).numpy()
            # Fourier Loss
            F_loss = pixelwise_loss(F_true, F_pred).numpy()
            # SSIM and PSNR
            saving_metric = SavingMetric(y_true, y_pred).numpy()
        else:
            pixel_loss = pixelwise_loss(y_true, y_pred)
            # Fourier Loss
            F_loss = pixelwise_loss(F_true, F_pred)
            # SSIM and PSNR
            saving_metric = SavingMetric(y_true, y_pred)
        F_loss = tf.cast(F_loss, dtype=tf.float32)
        loss = B1*pixel_loss + B2*F_loss + B3*saving_metric
        return loss
    loss_func.__name__ = name
    return loss_func