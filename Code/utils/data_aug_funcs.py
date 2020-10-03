# Customary Imports:
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import PIL
import imageio
import tensorflow.keras.backend as K
import utils.data_aug_funcs
##################################################################################################################################
'''
DATA AUGMENTATION FUNCTIONS:
'''
##################################################################################################################################
def normalize(tensor):
    # Normalizes Tensor from 0-1
    return tf.math.divide_no_nan(tf.math.subtract(tensor, tf.math.reduce_min(tensor)), 
                                 tf.math.subtract(tf.math.reduce_max(tensor), tf.math.reduce_min(tensor)))

def preprocess_function(image, prob = 0.25, max_shift = 0.10, lower_con_shift = 0.25, upper_con_shift = 1.75, lower = 10, upper = 30,
                        mean = 0.0, std_lower = 0.003, std_upper = 0.015, seed = 7):
    '''
    This function is the wrapper preprocess function to be used on all training data as a data
    augmentation step.
    '''
    img = normalize(image).numpy()
    img = add_rand_bright_shift(img, max_shift, prob, seed=seed)
    img = add_rand_contrast(img, lower_con_shift, upper_con_shift, prob, seed=seed)
    img = rand_adjust_jpeg_quality(img, lower, upper, prob/2, seed=seed)
    img = add_rand_gaussian_noise(img, mean, std_lower, std_upper, prob/2, seed=seed)
    '''
    img = tf.keras.preprocessing.image.random_zoom(img,zoom_range=(0.8,2),row_axis=0,col_axis=1,channel_axis=2,
                                                   fill_mode='nearest',cval=0.0,interpolation_order=3)
    '''
    #'''
    img[img > 1.0] = 1.0
    img[img < 0.0] = 0.0
    #'''
    #print(img.shape)
    return img
def preprocess_function_valtest(image):
    '''
    This function is the wrapper preprocess function to be used on all validation/test data.
    '''
    img = normalize(image)
    return img
def rand_adjust_jpeg_quality(batch, lower = 10, upper = 30, prob = 0.1, seed = None):
    '''
    This function randomly adjusts the image's jpeg compression quality with given probability.
    '''
    rand_var = tf.random.uniform(shape = [1], seed=seed).numpy()[0]
    batch_out = batch
    if rand_var < prob:
        batch_out = tf.image.random_jpeg_quality(batch, lower, upper, seed).numpy()
    return batch_out
def add_rand_gaussian_noise(batch, mean_val = 0.0, std_lower = 0.01, std_upper = 0.1, prob = 0.1, seed = None):
    '''
    This function introduces additive Gaussian Noise with a given mean and std, at
    a certain given probability.
    '''
    rand_var = tf.random.uniform(shape = [1], seed=seed).numpy()[0]
    batch_and_noise = batch
    if rand_var < prob:
        std = tf.random.uniform(shape = [1], minval=std_lower, 
                                maxval=std_upper, seed=seed).numpy()[0]
        noise = tf.random.normal(shape=tf.shape(batch), mean=mean_val, 
                                 stddev=std, dtype=tf.float32, seed=seed)
        batch_and_noise = tf.math.add(batch, noise).numpy()
        batch_and_noise[batch_and_noise > 1.0] = 1.0
        batch_and_noise[batch_and_noise < 0.0] = 0.0
    return batch_and_noise
def add_rand_bright_shift(batch, max_shift = 0.12, prob = 0.1, seed=None):
    '''
    Equivalent to adjust_brightness() using a delta randomly
    picked in the interval [-max_delta, max_delta) with a
    given probability that this function is performed on an image.
    The pixels lower than 0 are clipped to 0 and the pixels higher 
    than 1 are clipped to 1.
    '''
    rand_var = tf.random.uniform(shape = [1], seed=seed).numpy()[0]
    batch_and_bright_shift = batch
    if rand_var < prob:
        if len(batch.shape) == 2:
            batch_and_bright_shift = np.stack((batch_and_bright_shift,)*3, axis=-1)
        batch_and_bright_shift = tf.image.random_brightness(image=batch_and_bright_shift, 
                                                            max_delta=max_shift, 
                                                            seed=seed).numpy()
        batch_and_bright_shift[batch_and_bright_shift > 1.0] = 1.0
        batch_and_bright_shift[batch_and_bright_shift < 0.0] = 0.0
        if len(batch.shape) == 2:
            batch_and_bright_shift = batch_and_bright_shift[...,0]
    return batch_and_bright_shift
def add_rand_contrast(batch, lower = 0.2, upper = 1.8, prob = 0.1, seed=None):
    '''
    For each channel, this Op computes the mean of the image pixels in the channel 
    and then adjusts each component x of each pixel to (x - mean) * contrast_factor + mean
    with a given probability that this function is performed on an image. The pixels lower
    than 0 are clipped to 0 and the pixels higher than 1 are clipped to 1.
    '''
    rand_var = tf.random.uniform(shape = [1],seed=seed).numpy()[0]
    batch_and_rand_contrast = batch
    if rand_var < prob:
        if len(batch.shape) == 2:
            batch_and_rand_contrast = np.stack((batch_and_rand_contrast,)*3, axis=-1)
        batch_and_rand_contrast = tf.image.random_contrast(image=batch_and_rand_contrast, 
                                                           lower=lower, 
                                                           upper=upper, 
                                                           seed=seed).numpy()
        batch_and_rand_contrast[batch_and_rand_contrast > 1.0] = 1.0
        batch_and_rand_contrast[batch_and_rand_contrast < 0.0] = 0.0
        if len(batch.shape) == 2:
            batch_and_rand_contrast = batch_and_rand_contrast[...,0]
    return batch_and_rand_contrast
##################################################################################################################################
'''
UNUSED FUNCTIONS:
'''
##################################################################################################################################
def img_quantize(batch, prob, seed=None):
    rand_var = tf.random.uniform(shape = [1], seed=seed).numpy()[0]
    img = batch.copy()
    if tf.is_tensor(img):
        img = img.numpy()
    if rand_var < prob:
        img = exposure.rescale_intensity(img, in_range = 'image', 
                                         out_range = (0.0,255.0)).astype('uint8')
        img = exposure.rescale_intensity(img.astype('float32'), in_range = 'image', 
                                         out_range = (0.0,1.0))
    return img
def add_speckle_noise(batch, mean_val = 0.0, std_lower = 0.01, std_upper = 0.1, prob = 0.1, seed = None):
    '''
    This function introduces Speckle Noise with a given mean and std, at
    a certain given probability.
    '''
    rand_var = tf.random.uniform(shape = [1], seed=seed).numpy()[0]
    batch_and_noise = batch
    if rand_var < prob:
        std = tf.random.uniform(shape = [1], minval=std_lower, 
                                maxval=std_upper, seed=seed).numpy()[0]
        noise = tf.random.normal(shape=tf.shape(batch), mean=mean_val, 
                                 stddev=std, dtype=tf.float32, seed=seed)
        batch_and_noise = tf.math.add(batch, tf.math.multiply(batch, noise)).numpy()
        batch_and_noise = exposure.rescale_intensity(batch_and_noise, 
                                                     in_range='image', 
                                                     out_range=(0.0,1.0))
    return batch_and_noise