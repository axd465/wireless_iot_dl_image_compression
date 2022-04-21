# Customary Imports:
import numpy as np
import tensorflow as tf
###############################################################################
'''
JPEG UTILS:
'''
###############################################################################

def jpeg_encode_decode(img, quality):
    return tf.image.adjust_jpeg_quality(img, 
                                        quality, 
                                        name="jpeg_encode_decode")