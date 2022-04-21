# Customary Imports:
import tensorflow as tf
from tensorflow.image import random_flip_left_right as random_flip_lr
from tensorflow.image import random_flip_up_down as random_flip_ud
from functools import wraps

###############################################################################
'''
AUGMENTATION UTILS:
'''
###############################################################################
# General Augmentation Functions:


@tf.function(experimental_relax_shapes=True)
def normalize(tensor, clip=False,
              out_range=[0, 1],
              if_uint8=False):
    # Normalizes Tensor from 0-1
    out = tf.cast(tensor, tf.float32)
    if clip:
        out = tf.clip_by_value(out,
                               out_range[0],
                               out_range[1])
    elif if_uint8:
        out /= 255
        out *= (out_range[1] - out_range[0])
        out += out_range[0]
    else:
        out = tf.math.divide_no_nan(tf.math.subtract(out,
                                                     tf.math.reduce_min(out)),
                                    tf.math.subtract(tf.math.reduce_max(out),
                                                     tf.math.reduce_min(out)))
        out *= (out_range[1] - out_range[0])
        out += out_range[0]
    return out

@tf.function(experimental_relax_shapes=True)
def add_rand_gaussian_noise(x, mean_val=0.0, std_lower=0.001,
                            std_upper=0.005, prob=0.1, out_range=[0, 1],
                            seed=None):
    '''
    This function introduces additive Gaussian Noise
    with a given mean and std, at
    a certain given probability.
    '''
    rand_var = tf.random.uniform(shape=(), seed=seed)
    if rand_var < prob:
        std = tf.random.uniform(shape=(), minval=std_lower,
                                          maxval=std_upper, seed=seed)
        noise = tf.random.normal(shape=x.shape, mean=mean_val,
                                 stddev=std, dtype=tf.float32,
                                 seed=seed)
        x = tf.math.add(x, noise)
        x = tf.clip_by_value(x, out_range[0], out_range[1])
    return x


@tf.function
def add_rand_bright_shift(x, max_shift=0.12, prob=0.1,
                          out_range=[0, 1], seed=None):
    '''
    Equivalent to adjust_brightness() using a delta randomly
    picked in the interval [-max_delta, max_delta) with a
    given probability that this function is performed on an image.
    The pixels lower than 0 are clipped to 0 and the pixels higher
    than 1 are clipped to 1.
    '''
    rand_var = tf.random.uniform(shape=(), seed=seed)
    if rand_var < prob:
        x = tf.image.random_brightness(image=x,
                                       max_delta=max_shift,
                                       seed=seed)
        x = tf.clip_by_value(x, out_range[0], out_range[1])
    return x


@tf.function
def add_rand_contrast(x, lower=0.2, upper=1.8,
                      prob=0.1, out_range=[0, 1],
                      seed=None):
    '''
    For each channel, this Op computes the mean
    of the image pixels in the channel
    and then adjusts each component x of each pixel
    to (x - mean) * contrast_factor + mean
    with a given probability that this function is
    performed on an image. The pixels lower
    than 0 are clipped to 0 and the pixels higher
    than 1 are clipped to 1.
    '''
    rand_var = tf.random.uniform(shape=(), seed=seed)
    if rand_var < prob:
        x = tf.image.random_contrast(image=x, lower=lower,
                                     upper=upper, seed=seed)
        x = tf.clip_by_value(x, out_range[0], out_range[1])
    return x


@tf.function
def random_hue(x, max_delta=0.1, prob=0.1, out_range=[0, 1], seed=None):
    rand_var = tf.random.uniform(shape=(), seed=seed)
    if rand_var < prob:
        x = tf.image.random_hue(image=x, 
                                max_delta=max_delta, 
                                seed=seed)
        x = tf.clip_by_value(x, out_range[0], out_range[1])
    return x


@tf.function
def random_jpeg_quality(x, jpeg_quality_range=[70, 100], prob=0.1, seed=None):
    rand_var = tf.random.uniform(shape=(), seed=seed)
    if rand_var < prob:
        x = tf.image.random_jpeg_quality(x, 
                                         jpeg_quality_range[0], 
                                         jpeg_quality_range[1], 
                                         seed=seed)
    return x


@tf.function
def random_saturation(x, sat_factor_range=[0.8, 1.2], 
                      prob=0.1, out_range=[0, 1], seed=None):
    rand_var = tf.random.uniform(shape=(), seed=seed)
    if rand_var < prob:
        x = tf.image.random_saturation(x, 
                                       sat_factor_range[0], 
                                       sat_factor_range[1], 
                                       seed=seed)
        x = tf.clip_by_value(x, out_range[0], out_range[1])
    return x


@tf.function
def augment(ds, prob=1/3, 
            max_shift=0.10,
            con_factor_range=[0.2, 1.8], 
            max_hue_delta=0.1, 
            jpeg_quality_range=[70, 100],
            sat_factor_range=[0.8, 1.2],
            seed=7,
            out_range=[0, 1]):
    x = ds
    if tf.random.uniform(shape=(), seed=seed) > 0.5:
        x = random_flip_lr(x, seed=seed)
    if tf.random.uniform(shape=(), seed=seed) > 0.5:
        x = random_flip_ud(x, seed=seed)
    x = add_rand_bright_shift(x, max_shift, prob, out_range, seed=seed)
    x = add_rand_contrast(x, 
                          con_factor_range[0],
                          con_factor_range[1],
                          prob, out_range,
                          seed=seed)
    x = random_hue(x, 
                   max_hue_delta, 
                   prob, 
                   out_range,
                   seed=seed)
    # x = random_jpeg_quality(x, 
    #                         jpeg_quality_range, 
    #                         prob,
    #                         seed=seed)
    x = random_saturation(x, 
                          sat_factor_range, 
                          prob, 
                          out_range,
                          seed=seed)
    return ds


@tf.function(experimental_relax_shapes=True)
def augment_noise(ds, prob=1/3, mean=0.0, std_lower=0.003, std_upper=0.015,
                  seed=7,
                  out_range=[0, 1]):
    return add_rand_gaussian_noise(ds, mean, std_lower,
                                   std_upper, prob/4,
                                   out_range,
                                   seed=seed)


def keras_augment(ds, fill_mode='constant', 
                  interpolation_order=0,
                  rg=20, zoom_range=0.2, 
                  max_shift_xy=[0.1, 0.1], 
                  intensity_range=0.1, 
                  shear_intensity=0.2,
                  out_range=[0, 1]):
    
    x = tf.keras.preprocessing.image.random_rotation(ds, rg, 
                                                     row_axis=0, 
                                                     col_axis=1, 
                                                     channel_axis=2, 
                                                     fill_mode=fill_mode,
                                                     cval=0.0, 
                                                     interpolation_order=interpolation_order)
    x = tf.keras.preprocessing.image.random_zoom(x, zoom_range, 
                                                 row_axis=0, 
                                                 col_axis=1, 
                                                 channel_axis=2, 
                                                 fill_mode=fill_mode, 
                                                 cval=0.0, 
                                                 interpolation_order=interpolation_order)
    x = tf.keras.preprocessing.image.random_shift(x, max_shift_xy[0],
                                                  max_shift_xy[1],
                                                  row_axis=0, col_axis=1,
                                                  channel_axis=2,
                                                  fill_mode=fill_mode,
                                                  interpolation_order=interpolation_order)
    x = tf.keras.preprocessing.image.random_channel_shift(x, 
                                                          intensity_range, 
                                                          channel_axis=2)
    x = tf.keras.preprocessing.image.random_shear(x, shear_intensity, 
                                                  row_axis=0, 
                                                  col_axis=1, 
                                                  channel_axis=2, 
                                                  fill_mode=fill_mode, 
                                                  cval=0.0, 
                                                  interpolation_order=interpolation_order)
    x = tf.clip_by_value(x, out_range[0], out_range[1])
    return x