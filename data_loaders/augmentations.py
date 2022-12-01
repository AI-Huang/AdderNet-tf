#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-11-22 14:59
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import tensorflow as tf
from tensorflow.keras import layers

minor_version = int(tf.__version__.split('.')[1])
if minor_version >= 6:  # TensorFlow 2.6
    from tensorflow.keras.layers import Rescaling
else:
    from tensorflow.keras.layers.experimental.preprocessing import Rescaling
try:
    from tensorflow.keras.layers import RandomFlip, RandomCrop
except:
    from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomCrop


def img_rescale(x):
    return x / 255.0


def img_rescale_lambda():
    return layers.Lambda(lambda x: img_rescale(x))


to_tensor = img_rescale_lambda()
to_tensor = tf.keras.Sequential([
    to_tensor
])

# Optional
tf_rescale = Rescaling(1./255)

# Pad, flip, and crop
pad_and_crop = tf.keras.Sequential([
    layers.ZeroPadding2D(padding=(4, 4)),
    RandomFlip("horizontal"),
    RandomCrop(32, 32)
])
