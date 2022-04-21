# Customary Imports:
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Add, Input
from tensorflow.keras.layers import BatchNormalization, UpSampling2D, Concatenate, Conv2DTranspose

###################################################################################################
'''
MODEL DEFINITION:
Simple Autoencoder
'''

def AutoEncoder(img, filters=32, kernel_size=3, padding='same',
                activation='relu', kernel_initializer='glorot_normal',
                kernel_regularizer=tf.keras.regularizers.L2(0.000001)):
    shortcut1_1 = img
    out = DownBlock(img, filters, kernel_size, padding, activation, kernel_initializer)
    out = DownBlock(out, filters*2, kernel_size, padding, activation, kernel_initializer)
    out = DownBlock(out, filters*2*2, kernel_size, padding, activation, kernel_initializer)
    out = DownBlock(out, filters*2*2*2, kernel_size, padding, activation, kernel_initializer)


    out = BridgeBlock(out, filters*2*2*2*2, kernel_size, padding, activation, kernel_initializer)

    out = UpBlock(out, filters*2*2*2, kernel_size, padding, activation, kernel_initializer)
    out = UpBlock(out, filters*2*2, kernel_size, padding, activation, kernel_initializer)
    out = UpBlock(out, filters*2, kernel_size, padding, activation, kernel_initializer)

    out = Conv2D_BatchNorm(out, filters, kernel_size, strides=1, padding=padding,
                           activation=activation, kernel_initializer=kernel_initializer)

    out = Conv2D_BatchNorm(out, filters, kernel_size, strides=1, padding=padding,
                           activation=activation, kernel_initializer=kernel_initializer)

    out = Conv2D_BatchNorm(out, filters=3, kernel_size=1, strides=1, padding=padding, 
                           activation='linear', kernel_initializer=kernel_initializer)

    return out

def DownBlock(img, filters, kernel_size, padding, activation, kernel_initializer):
    #print('DOWN_in: '+str(input.shape))
    out = Conv2D_BatchNorm(img, filters, kernel_size, strides=1, padding=padding,
                           activation=activation, kernel_initializer=kernel_initializer)
    out = Conv2D_BatchNorm(out, filters, kernel_size, strides=2, padding=padding,
                           activation=activation, kernel_initializer=kernel_initializer)
    # out = MaxPooling2D(pool_size = (2, 2))(out)
    #print('DOWN_out: '+str(out.shape))
    return out

def BridgeBlock(img, filters, kernel_size, padding, activation, kernel_initializer):
    #print('UP_in: '+str(input.shape))
    #print(filters)
    out = Conv2D_BatchNorm(img, filters, kernel_size, strides = 1, padding=padding,
                           activation=activation, kernel_initializer = kernel_initializer)
    out = Conv2D_BatchNorm(out, filters, kernel_size, strides = 1, padding=padding,
                           activation=activation, kernel_initializer = kernel_initializer)
    out = UpSampling2D(size = (2, 2), interpolation='bilinear')(out)
    #print('UP_out: '+str(out.shape))
    return out

def UpBlock(img, filters, kernel_size, padding, activation, kernel_initializer):
    #print('UP_in: '+str(input.shape))
    #print(filters)
    out = Conv2D_BatchNorm(img, filters, kernel_size, strides = 1, padding=padding,
                           activation=activation, kernel_initializer = kernel_initializer)
    out = Conv2D_BatchNorm(out, filters, kernel_size, strides = 1, padding=padding,
                           activation=activation, kernel_initializer = kernel_initializer)
    out = UpSampling2D(size = (2, 2), interpolation='bilinear')(out)
    #print('UP_out: '+str(out.shape))
    return out

###################################################################################################
'''
MODEL FUNCTIONS:
'''
def Conv2D_BatchNorm(img, filters, kernel_size=3, strides=1, padding='same',
                     activation='linear', kernel_initializer='glorot_normal'):
    out = Conv2D(filters=filters, kernel_size=kernel_size,
                 strides=strides, padding=padding,
                 activation=activation,
                 kernel_initializer=kernel_initializer)(img)
    out = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
                             scale=True, beta_initializer='zeros', gamma_initializer='ones',
                             moving_mean_initializer='zeros', moving_variance_initializer='ones',
                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                             gamma_constraint=None)(out)
    return out

def Conv2D_Transpose_BatchNorm(img, filters, kernel_size=3, strides=2, padding='same',
                               activation='relu', kernel_initializer='glorot_normal'):
    # Conv2DTranspose also known as a 2D Deconvolution
    out = Conv2DTranspose(filters, kernel_size, strides=2, padding=padding,
                          activation=activation, kernel_initializer=kernel_initializer)(img)
    out = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
                             scale=True, beta_initializer='zeros', gamma_initializer='ones',
                             moving_mean_initializer='zeros', moving_variance_initializer='ones',
                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                             gamma_constraint=None)(out)
    return out

###################################################################################################
'''
FUNCTION TO INSTANTIATE MODEL:
'''
def getModel(input_shape, filters, kernel_size, padding='same', activation='relu',
             kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.L2(0.000001)):
    model_inputs = Input(shape=input_shape, name='img')
    model_outputs = AutoEncoder(model_inputs, filters=filters,
                                kernel_size=kernel_size, padding=padding,
                                activation=activation,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)
    model = Model(model_inputs, model_outputs, name='AutoEncoder')
    return model
getModel.__name__ = 'AutoEncoder'
