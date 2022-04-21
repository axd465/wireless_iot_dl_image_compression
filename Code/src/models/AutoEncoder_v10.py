# Customary Imports:
import tensorflow as tf
import os
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Add, \
                                    BatchNormalization, UpSampling2D, \
                                    Dropout, AlphaDropout, AveragePooling2D, \
                                    Input, Concatenate, Conv2DTranspose, \
                                    SpatialDropout2D
###################################################################################################
'''
MODEL DEFINITION:
Simple Autoencoder
'''
def AutoEncoder(img,
                filters=32,
                kernel_size=3,
                activation='relu',
                prob=0,
                padding='same',
                kernel_initializer='glorot_normal',
                kernel_regularizer=None,
                activity_regularizer=None,
                dilation_rate=1,
                strided_conv=False):
    conv_args = {
        'filters': filters,
        'kernel_size': kernel_size,
        'activation': activation,
        'prob': prob,
        'padding': padding,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'activity_regularizer': activity_regularizer,
        'dilation_rate': dilation_rate,
    }

    out = DownBlock(img, strided_conv, **conv_args)
    conv_args['filters'] *= 2
    out = DownBlock(out, strided_conv, **conv_args)
    conv_args['filters'] *= 2
    out = DownBlock(out, strided_conv, **conv_args)
    conv_args['filters'] *= 2
    out = DownBlock(out, strided_conv, **conv_args)
    conv_args['filters'] *= 2
    out = DownBlock(out, strided_conv, **conv_args)
    # conv_args['filters'] *= 2
    # out = DownBlock(out, strided_conv, **conv_args)
    # conv_args['filters'] *= 2
    # out = DownBlock(out, strided_conv, **conv_args)

    conv_args['filters'] *= 2
    out = Conv2D_BatchNorm(out, **conv_args)
    out = Conv2D_BatchNorm(out, **conv_args)
    out = BridgeBlock(out, **conv_args)

    # conv_args['filters'] /= 2
    # out = UpBlock(out, **conv_args)
    # conv_args['filters'] /= 2
    # out = UpBlock(out, **conv_args)
    conv_args['filters'] /= 2
    out = UpBlock(out, **conv_args)
    conv_args['filters'] /= 2
    out = UpBlock(out, **conv_args)
    conv_args['filters'] /= 2
    out = UpBlock(out, **conv_args)
    conv_args['filters'] /= 2
    out = UpBlock(out, **conv_args)
    conv_args['filters'] /= 2

    out = Conv2D_BatchNorm(out, **conv_args)

    out = Conv2D_BatchNorm(out, **conv_args)

    # 1x1 Convolution Followed by Identity as Activation:
    conv_args['filters'] = 3
    conv_args['kernel_size'] = 1
    conv_args['activation'] = 'linear'
    out = Conv2D_BatchNorm(out, **conv_args)
    out = Normalize(out)
    # out = Conv2D(**conv_args)(out)
    # out = Add()([out, shortcut1_1])
    # out = BatchNormalization()(out)

    return out


def DownBlock(img, strided_conv, **conv_args):
    #print('DOWN_in: '+str(img.shape))
    out = Conv2D_BatchNorm(img, **conv_args)
    out = Conv2D_BatchNorm(out, **conv_args)
    kwargs = conv_args.copy()
    if strided_conv:
        kwargs['strides'] = 2
        kwargs['filters'] *= 2
        out = Conv2D_BatchNorm(out, **kwargs)
    else:
        # kwargs['kernel_size'] = 1
        out = MaxPooling2D(pool_size=(2, 2))(out)
        # out = Conv2D_BatchNorm(out, **kwargs)
    #print('DOWN_out: '+str(out.shape))
    return out


def BridgeBlock(img, **conv_args):
    #print('UP_in: '+str(img.shape))
    #print(filters)
    out = Conv2D_BatchNorm(img, **conv_args)
    out = Conv2D_BatchNorm(out, **conv_args)
    out = Normalize(out)
    # kwargs = conv_args.copy()
    # kwargs['filters'] //= 2
    # kwargs['kernel_size'] = 1
    # out = UpSampling2D(size=(2, 2),
    #                    interpolation='bilinear')(out)
    # out = Conv2D_BatchNorm(out, **kwargs)
    #print('UP_out: '+str(out.shape))
    kwargs = conv_args.copy()
    kwargs['kernel_size'] = 1
    # out = Conv2D_BatchNorm(out, **kwargs)
    out = Pixel_Shuffle(out, upscale_factor=2)
    # out = Conv2D_BatchNorm(out, **kwargs)
    return out


def UpBlock(img, **conv_args):
    #print('UP_in: '+str(img.shape))
    #print(filters)
    out = Conv2D_BatchNorm(img, **conv_args)
    out = Conv2D_BatchNorm(out, **conv_args)
    # kwargs = conv_args.copy()
    # kwargs['filters'] //= 2
    # kwargs['kernel_size'] = 1
    # out = UpSampling2D(size=(2, 2),
    #                    interpolation='bilinear')(out)
    # out = Conv2D_BatchNorm(out, **kwargs)
    #print('UP_out: '+str(out.shape))
    kwargs = conv_args.copy()
    kwargs['kernel_size'] = 1
    # out = Conv2D_BatchNorm(out, **kwargs)
    out = Pixel_Shuffle(out, upscale_factor=2)
    # out = Conv2D_BatchNorm(out, **kwargs)
    return out

###################################################################################################
'''
MODEL FUNCTIONS:
'''


def Conv2D_BatchNorm(img, prob=0, **conv_args):
    out = Conv2D(**conv_args)(img)
    if prob is not None:
        out = SpatialDropout2D(prob)(out)
    out = BatchNormalization()(out)
    return out


# @tf.function(experimental_relax_shapes=True)
def Normalize(img):
    min_val = tf.math.reduce_min(img, axis=[1, 2, 3], keepdims=True)
    max_val = tf.math.reduce_max(img, axis=[1, 2, 3], keepdims=True)
    out = tf.math.divide_no_nan(tf.math.subtract(img,
                                                 min_val),
                                tf.math.subtract(max_val,
                                                 min_val))
    return out

# See https://keras.io/examples/vision/super_resolution_sub_pixel/#build-a-model
# for source of Subpixel CNN definition/code
def Pixel_Shuffle(img, upscale_factor=2):
    out = tf.nn.depth_to_space(img, upscale_factor)
    return out

def Conv2D_Transpose_BatchNorm(img, prob=0.0, **conv_args):
    # Conv2DTranspose also known as a 2D Deconvolution
    out = Conv2DTranspose(**conv_args)(img)
    if prob is not None:
        out = SpatialDropout2D(prob)(out)
    out = BatchNormalization()(out)
    return out

###################################################################################################
'''
FUNCTION TO INSTANTIATE MODEL:
'''

def getModel(input_shape, **kwargs):
    accept_kwargs = ['filters',
                     'kernel_size',
                     'activation',
                     'prob',
                     'padding',
                     'kernel_initializer',
                     'kernel_regularizer',
                     'dilation_rate',
                     'strided_conv']
    model_inputs = Input(shape=input_shape, name='img')
    model_kwargs = {i:kwargs[i] for i in kwargs \
                    if i in accept_kwargs}
    model_outputs = AutoEncoder(model_inputs,
                                **model_kwargs)
    model = Model(model_inputs, model_outputs, name='AutoEncoder')
    return model
getModel.__name__ = 'AutoEncoder'
