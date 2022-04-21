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


# @tf.function(experimental_relax_shapes=True)
# def PSNR(y_true, y_pred):
#     max_pixel = 1.0
#     clip = True
#     map_norm = partial(normalize, clip=clip)
#     y_true_norm = tf.map_fn(map_norm, y_true)
#     y_pred_norm = tf.map_fn(map_norm, y_pred)
#     PSNR = tf.image.psnr(y_true_norm, y_pred_norm, max_pixel)
#     return PSNR


# @tf.function(experimental_relax_shapes=True)
# def SSIM(y_true, y_pred):
#     max_pixel = 1.0
#     clip = True
#     map_norm = partial(normalize, clip=clip)
#     y_true_norm = tf.map_fn(map_norm, y_true)
#     y_pred_norm = tf.map_fn(map_norm, y_pred)
#     SSIM = tf.image.ssim(y_true_norm, y_pred_norm, max_pixel, filter_size=11,
#                          filter_sigma=1.5, k1=0.01, k2=0.03)
#     return SSIM

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


@tf.function(experimental_relax_shapes=True)
def SSIM_1D(y_true, y_pred, **kwargs):
    max_pixel = 1.0
    # clip = True
    # map_norm = partial(normalize, clip=clip)
    # y_true_norm = tf.map_fn(map_norm, y_true)
    # y_pred_norm = tf.map_fn(map_norm, y_pred)
    SSIM = tf_ssim(y_true_norm, y_pred_norm,
                   max_pixel,
                   filter_size=[11, 1],
                   filter_sigma=1.5,
                   k1=0.01, k2=0.03,
                   **kwargs)
    return SSIM


@tf.function(experimental_relax_shapes=True)
def MS_SSIM(y_true, y_pred):
    max_pixel = 1.0
    # clip = True
    # map_norm = partial(normalize, clip=clip)
    # y_true_norm = tf.map_fn(map_norm, y_true)
    # y_pred_norm = tf.map_fn(map_norm, y_pred)
    MS_SSIM = tf.image.ssim_multiscale(y_true_norm,
                                       y_pred_norm,
                                       max_pixel)
    return MS_SSIM


@tf.function(experimental_relax_shapes=True)
def quality_loss(y_true, y_pred, A1=75, A2=400, input_type='2D'):
    # Combines Insight from SSIM and PSNR
    if input_type == '2D':
        ssim = SSIM(y_true, y_pred)
    else:
        ssim = SSIM_1D(y_true, y_pred)
    psnr = PSNR(y_true, y_pred)
    # Normalize for Minimization:
    ssim_norm = 1 - ssim
    psnr_norm = 0
    psnr_norm = tf.clip_by_value((A1 - psnr)/A2, 0, tf.float32.max)
    loss = ssim_norm + psnr_norm
    return loss


@tf.function(experimental_relax_shapes=True)
def downsample(y_true, y_pred, down_ratio=[5, 1]):

    down_y_true = y_true[:, ::down_ratio[0], ::down_ratio[1], :]
    down_y_pred = y_pred[:, ::down_ratio[0], ::down_ratio[1], :]
    return down_y_true, down_y_pred


@tf.function(experimental_relax_shapes=True)
def down_consistency_loss(y_true, y_pred, down_ratio=[5, 1],
                          pixelwise_loss='MAE'):

    down_y_true, down_y_pred = downsample(y_true,
                                          y_pred,
                                          down_ratio)
    if pixelwise_loss == 'MSE':
        pixelwise_loss = tf.keras.losses.MeanSquaredError()
    else:
        pixelwise_loss = tf.keras.losses.MeanAbsoluteError()
    loss = pixelwise_loss(down_y_true, down_y_pred)

    return loss

# # Model Loss Function (Fourier Loss):
# @tf.function(experimental_relax_shapes=True)
# def rfft(img):
#     # RFFT Function to be performed for each instance in batch
#     return tf.signal.rfft(img)


# Model Loss Function (Fourier Transform 2D):
@tf.function(experimental_relax_shapes=True)
def fft2d_abs(img):
    out = tf.transpose(img, perm=[0, 3, 1, 2])
    out = tf.signal.fft2d(tf.complex(out, tf.zeros_like(out)))
    out = tf.math.abs(out)
    out = tf.transpose(out, perm=[0, 2, 3, 1])
    return out


# Model Loss Function (Fourier Transform 2D):
@tf.function(experimental_relax_shapes=True)
def fft2d_abs_weighted(img, sigma=85.0):
    #in_batch, in_i, in_j, in_channel = tf.shape(img)
    in_batch, in_i, in_j, in_channel = (None, 512, 1, 1)
    out = tf.transpose(img, perm=[0, 3, 1, 2])
    out = tf.signal.fft2d(tf.complex(out, tf.zeros_like(out)))
    out = tf.math.abs(out)
    out = tf.transpose(out, perm=[0, 2, 3, 1])
    weight = _get_tf_gauss_kernel((in_i, in_j), sigma)
    weight = tf.expand_dims(weight, -1)
    weight = tf.expand_dims(weight, 0)
    out = tf.math.multiply(out, weight)
    return out


@tf.function(experimental_relax_shapes=True)
def _get_tf_gauss_kernel(shape=(512, 1), sigma=85.0):
    """build the gaussain filter"""
    m, n = [(ss-1.)/2. for ss in shape]
    x = tf.expand_dims(tf.range(-n, n+1, dtype=tf.float32), 1)
    y = tf.expand_dims(tf.range(-m, m+1, dtype=tf.float32), 0)
    h = tf.exp(tf.math.divide_no_nan(-((x**2) + (y**2)), 2*(sigma**2)))
    h = tf.math.divide_no_nan(h, tf.reduce_sum(h))
    return h


def dct_2d(
        img,
        norm=None # can also be 'ortho'
):
    img = tf.transpose(img, perm=[0, 3, 1, 2])
    X1 = tf.signal.dct(img, type=2, norm=norm)
    X1_t = tf.transpose(X1, perm=[0, 1, 3, 2])
    X2 = tf.signal.dct(X1_t, type=2, norm=norm)
    X2_t = tf.transpose(X2, perm=[0, 1, 3, 2])
    out = tf.transpose(X2_t, perm=[0, 2, 3, 1])
    return out

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
# # Experimental Model Loss Function:
# def model_loss(B1=1.0, B2=0.000075, B3=0.0,
#                B4=0.0, B5=0.0, B6=0.0,
#                down_ratio=[5, 1], mse=False,
#                name='model_loss', weighted_fft=False,
#                **kwargs):
#     @tf.function(experimental_relax_shapes=True)
#     def loss_func(y_true, y_pred):
#         pixel_loss = 0
#         F_loss = 0
#         Q_loss = 0
#         down_loss = 0
#         max_loss = 0
#         min_loss = 0
#         if mse:
#             pixelwise_loss = tf.keras.losses.MeanSquaredError()
#         else:
#             pixelwise_loss = tf.keras.losses.MeanAbsoluteError()
#         if B1 > 0:
#             pixel_loss = pixelwise_loss(y_true, y_pred)
#         # Fourier Loss
#         if B2 > 0:
#             if weighted_fft:
#                 F_true = fft2d_abs_weighted(y_true)
#                 F_pred = fft2d_abs_weighted(y_pred)
#             else:
#                 F_true = fft2d_abs(y_true)
#                 F_pred = fft2d_abs(y_pred)
#             F_loss = pixelwise_loss(F_true, F_pred)
#         # SSIM and PSNR
#         if B3 > 0:
#             Q_loss = quality_loss(y_true, y_pred, **kwargs)
#         # Downsample Consistency Loss
#         if B4 > 0:
#             down_y_true, down_y_pred = downsample(y_true,
#                                                   y_pred,
#                                                   down_ratio)
#             down_loss = pixelwise_loss(down_y_true, down_y_pred)
#         # Peak Loss
#         if B5 > 0:
#             y_true_max = tf.math.reduce_max(y_true, axis=1, keepdims=True)
#             y_pred_max = tf.math.reduce_max(y_pred, axis=1, keepdims=True)
#             max_loss = pixelwise_loss(y_true_max, y_pred_max)
#         if B6 > 0:
#             y_true_min = tf.math.reduce_min(y_true, axis=1, keepdims=True)
#             y_pred_min = tf.math.reduce_min(y_pred, axis=1, keepdims=True)
#             min_loss = pixelwise_loss(y_true_min, y_pred_min)
#         return (B1*pixel_loss + B2*F_loss +
#                 B3*Q_loss + B4*down_loss +
#                 B5*max_loss +
#                 B6*min_loss)
#     loss_func.__name__ = name
#     return loss_func


# # Experimental Model Loss Function:
# def model_loss(B1=1.0, B2=0.000075, B3=0.0, mse=True, name='model_loss'):
#     @tf.function(experimental_relax_shapes=True)
#     def loss_func(y_true, y_pred):
#         pixel_loss = 0
#         F_loss = 0
#         Q_loss = 0
#         if B1 > 0:
#             if mse:
#                 pixelwise_loss = tf.keras.losses.MeanSquaredError()
#             else:
#                 pixelwise_loss = tf.keras.losses.MeanAbsoluteError()
#             pixel_loss = pixelwise_loss(y_true, y_pred)
#         # Fourier Loss
#         if B2 > 0:
#             F_true = fft2d_abs(y_true)
#             F_pred = fft2d_abs(y_pred)
#             F_loss = pixelwise_loss(F_true, F_pred)
#         # SSIM and PSNR
#         if B3 > 0:
#             Q_loss = quality_loss(y_true, y_pred)
#         return B1*pixel_loss + B2*F_loss + B3*Q_loss
#     loss_func.__name__ = name
#     return loss_func

# # Experimental Model Loss Function:
# def model_loss(B1=1.0, B2=0.000075, B3=0.0, mse=True, name='model_loss'):
#     @tf.function(experimental_relax_shapes=True)
#     def loss_func(y_true, y_pred):
#         pixel_loss = 0
#         F_loss = 0
#         Q_loss = 0
#         if B1>0:
#             if mse:
#                 pixelwise_loss = tf.keras.losses.MeanSquaredError()
#             else:
#                 pixelwise_loss = tf.keras.losses.MeanAbsoluteError()
#             pixel_loss = pixelwise_loss(y_true, y_pred)
#         # Fourier Loss
#         if B2>0:
#             F_true = tf.math.abs(rfft(y_true)) if mse else rfft(y_true)
#             F_pred = tf.math.abs(rfft(y_pred)) if mse else rfft(y_pred)
#             F_loss = pixelwise_loss(F_true, F_pred)
#         # SSIM and PSNR
#         if B3>0:
#             Q_loss = quality_loss(y_true, y_pred)
#         return B1*pixel_loss + B2*F_loss + B3*Q_loss
#     loss_func.__name__ = name
#     return loss_func

# # Experimental Model Loss Function:
# def model_loss(B1=1.0, B2=0.000075, B3=0.0, mse=True, name='model_loss'):
#     @tf.function(experimental_relax_shapes=True)
#     def loss_func(y_true, y_pred):
#         pixel_loss = 0
#         F_loss = 0
#         Q_loss = 0
#         if B1>0:
#             if mse:
#                 pixelwise_loss = tf.keras.losses.MeanSquaredError()
#             else:
#                 pixelwise_loss = tf.keras.losses.MeanAbsoluteError()
#             pixel_loss = pixelwise_loss(y_true, y_pred)
#         # Fourier Loss
#         if B2>0:
#             F_true = tf.math.abs(rfft(y_true))[:,1:,...] if mse else rfft(y_true)[:,1:,...]
#             F_pred = tf.math.abs(rfft(y_pred))[:,1:,...] if mse else rfft(y_pred)[:,1:,...]
#             F_loss = pixelwise_loss(F_true, F_pred)
#         # SSIM and PSNR
#         if B3>0:
#             Q_loss = quality_loss(y_true, y_pred)
#         return B1*pixel_loss + B2*F_loss + B3*Q_loss
#     loss_func.__name__ = name
#     return loss_func

# # Experimental Model Loss Function:
# def model_loss(B1=1.0, B2=0.000075, B3=0.001, mse=True, name='loss_func'):
#     #@tf.function
#     def loss_func(y_true, y_pred):
#         F_true = tf.map_fn(fft_mag, y_true)
#         F_pred = tf.map_fn(fft_mag, y_pred)
#         if mse:
#             pixelwise_loss = tf.keras.losses.MeanSquaredError()
#         else:
#             pixelwise_loss = tf.keras.losses.MeanAbsoluteError()
# #         if tf.executing_eagerly():
# #             pixel_loss = pixelwise_loss(y_true, y_pred).numpy()
# #             # Fourier Loss
# #             F_loss = pixelwise_loss(F_true, F_pred).numpy()
# #             # SSIM and PSNR
# #             saving_metric = quality_loss(y_true, y_pred).numpy()
# #         else:
# #             pixel_loss = pixelwise_loss(y_true, y_pred)
# #             # Fourier Loss
# #             F_loss = pixelwise_loss(F_true, F_pred)
# #             # SSIM and PSNR
# #             saving_metric = quality_loss(y_true, y_pred)
#         pixel_loss = pixelwise_loss(y_true, y_pred)
#         # Fourier Loss
#         F_loss = pixelwise_loss(F_true, F_pred)
#         # SSIM and PSNR
#         Q_loss = quality_loss(y_true, y_pred)
#         loss = B1*pixel_loss + B2*F_loss + B3*Q_loss
#         return loss
#     loss_func.__name__ = name
#     return loss_func


def _tf_fspecial_gauss_1D(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data = np.mgrid[-size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)

    g = tf.exp(-((x**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, max_pixel=1, filter_size=11, filter_sigma=1.5, 
            k1 = 0.01, k2 = 0.03, cs_map=False, mean_metric=True,
            mean_axis=1):
    # window shape [filter_size, 1]
    if isinstance(filter_size, list): 
        window = _tf_fspecial_gauss_1D(filter_size[0], 
                                       filter_sigma)
    # window shape [filter_size, size]
    else:
        window = _tf_fspecial_gauss(filter_size, 
                                    filter_sigma)
    C1 = (k1*max_pixel)**2
    C2 = (k2*max_pixel)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    else:
        value = tf.reduce_mean(value, axis=mean_axis)
    return value


# def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
#     weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
#     mssim = []
#     mcs = []
#     for l in range(level):
#         ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
#         mssim.append(tf.reduce_mean(ssim_map))
#         mcs.append(tf.reduce_mean(cs_map))
#         filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
#         filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
#         img1 = filtered_im1
#         img2 = filtered_im2

#     # list to tensor of dim D+1
#     mssim = tf.stack(mssim, axis=0)
#     mcs = tf.stack(mcs, axis=0)

#     value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
#                             (mssim[level-1]**weight[level-1]))

#     if mean_metric:
#         value = tf.reduce_mean(value)
#     return value