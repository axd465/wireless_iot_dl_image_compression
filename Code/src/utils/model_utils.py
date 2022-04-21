# Customary Imports:
import tensorflow as tf
import numpy as np
from functools import partial
import importlib
import utils.augmentation_utils
importlib.reload(utils.augmentation_utils)
from utils.augmentation_utils import normalize


###############################################################################
'''
MODEL UTILS:
'''
###############################################################################
# Custom Metrics:

@tf.function(experimental_relax_shapes=True)
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    # clip = True
    # map_norm = partial(normalize, clip=clip)
    # y_true_norm = tf.map_fn(map_norm, y_true)
    # y_pred_norm = tf.map_fn(map_norm, y_pred)
    PSNR = tf.image.psnr(y_true, y_pred, max_pixel)
    return PSNR


@tf.function(experimental_relax_shapes=True)
def SSIM(y_true, y_pred):
    max_pixel = 1.0
    # clip = True
    # map_norm = partial(normalize, clip=clip)
    # y_true_norm = tf.map_fn(map_norm, y_true)
    # y_pred_norm = tf.map_fn(map_norm, y_pred)
    SSIM = tf.image.ssim(y_true, y_pred, max_pixel, filter_size=11,
                         filter_sigma=1.5, k1=0.01, k2=0.03)
    return SSIM


# Experimental Model Loss Function:
def model_loss(B1=1.0, B2=0.001, 
               pixelwise_loss=tf.keras.losses.MeanAbsoluteError(),
               name='model_loss',
               **kwargs):
    @tf.function(experimental_relax_shapes=True)
    def loss_func(y_true, y_pred):
        pixel_loss = 0
        SSIM_loss = 0
        if B1 > 0:
            pixel_loss = pixelwise_loss(y_true, y_pred)
        # SSIM Loss
        if B2 > 0:
            SSIM_loss = 1 - tf.math.reduce_mean(SSIM(y_true, y_pred))
        return (B1*pixel_loss + B2*SSIM_loss)
    loss_func.__name__ = name
    return loss_func
