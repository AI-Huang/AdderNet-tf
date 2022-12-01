#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-09-21 21:35
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://github.com/huawei-noah/AdderNet/blob/master/adder.py
# @RefLink : https://github.com/tensorflow/tensorflow/blob/r2.3/tensorflow/python/keras/layers/convolutional.py#L512

"""Requirements
TensorFlow>=2.3
"""

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils


@tf.custom_gradient
def adder2d_im2col(X_col, W_col):
    """adder2d_im2col
    Inference function for Adder2D layer.
    # Arguments
        X: tf.Tensor, inputs with default shape (N_X, H, W, C_in).
        W: tf.Tensor, kernels with default shape (h_filter, w_filter, C_in, N_filters)
        stride:
        padding: "valid" or "same".

    # Returns
        outputs (tensor): outputs tensor as input to the next layer
    """

    # adder conv
    outputs = tf.abs((tf.expand_dims(W_col, 0)-tf.expand_dims(X_col, 2)))
    outputs = - tf.reduce_sum(outputs, 1)

    def adder2d_grad(upstream):
        grad_W_col = tf.reduce_sum(
            (tf.expand_dims(X_col, 2)-tf.expand_dims(W_col, 0))
            * tf.expand_dims(upstream, 1),
            0)
        grad_W_col = grad_W_col / \
            tf.clip_by_value(
                tf.norm(grad_W_col, ord=2), 1e-12, tf.float32.max) * \
            tf.sqrt(1.0*W_col.shape[1]*W_col.shape[0]) / 5.0

        grad_X_col = tf.reduce_sum(
            -tf.clip_by_value((tf.expand_dims(X_col, 2) -
                               tf.expand_dims(W_col, 0)),
                              -1, 1)
            * tf.expand_dims(upstream, 1),
            2)

        return grad_X_col, grad_W_col

    return outputs, adder2d_grad


class Adder2D(Layer):
    """2D convolution layer **using adder operation** (e.g. spatial convolution over images).
    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.
    Examples:
    >>> # The inputs are 28x28 RGB images with `channels_last` and the batch
    >>> # size is 4.
    >>> input_shape = (4, 28, 28, 3)
    >>> x = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.Adder2D(
    ... 2, 3, activation='relu', input_shape=input_shape[1:])(x)
    >>> print(y.shape)
    (4, 26, 26, 2)
    >>> # With `dilation_rate` as 2.
    >>> input_shape = (4, 28, 28, 3)
    >>> x = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.Adder2D(
    ... 2, 3, activation='relu', dilation_rate=2, input_shape=input_shape[1:])(x)
    >>> print(y.shape)
    (4, 24, 24, 2)
    >>> # With `padding` as "same".
    >>> input_shape = (4, 28, 28, 3)
    >>> x = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.Adder2D(
    ... 2, 3, activation='relu', padding="same", input_shape=input_shape[1:])(x)
    >>> print(y.shape)
    (4, 28, 28, 2)
    >>> # With extended batch shape [4, 7]:
    >>> input_shape = (4, 7, 28, 28, 3)
    >>> x = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.Adder2D(
    ... 2, 3, activation='relu', input_shape=input_shape[2:])(x)
    >>> print(y.shape)
    (4, 7, 26, 26, 2)
    Arguments:
        filters: Integer, the dimensionality of the output space (i.e. the number of
        output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the height
        and width of the 2D convolution window. Can be a single integer to specify
        the same value for all spatial dimensions.
        strides: An integer or tuple/list of 2 integers, specifying the strides of
        the convolution along the height and width. Can be a single integer to
        specify the same value for all spatial dimensions. Specifying any stride
        value != 1 is incompatible with specifying any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        `"valid"` means no padding. `"same"` results in padding evenly to
        the left/right or up/down of the input such that output has the same
        height/width dimension as the input.
        data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs. `channels_last` corresponds
        to inputs with shape `(batch_size, height, width, channels)` while
        `channels_first` corresponds to inputs with shape `(batch_size, channels,
        height, width)`. It defaults to the `image_data_format` value found in
        your Keras config file at `~/.keras/keras.json`. If you never set it, then
        it will be `channels_last`.
        dilation_rate: an integer or tuple/list of 2 integers, specifying the
        dilation rate to use for dilated convolution. Can be a single integer to
        specify the same value for all spatial dimensions. Currently, specifying
        any `dilation_rate` value != 1 is incompatible with specifying any stride
        value != 1.
        groups: A positive integer specifying the number of groups in which the
        input is split along the channel axis. Each group is convolved separately
        with `filters / groups` filters. The output is the concatenation of all
        the `groups` results along the channel axis. Input channels and `filters`
        must both be divisible by `groups`.
        activation: Activation function to use. If you don't specify anything, no
        activation is applied (see `keras.activations`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix (see
        `keras.initializers`).
        bias_initializer: Initializer for the bias vector (see
        `keras.initializers`).
        kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix (see `keras.regularizers`).
        bias_regularizer: Regularizer function applied to the bias vector (see
        `keras.regularizers`).
        activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation") (see `keras.regularizers`).
        kernel_constraint: Constraint function applied to the kernel matrix (see
        `keras.constraints`).
        bias_constraint: Constraint function applied to the bias vector (see
        `keras.constraints`).
    Input shape:
        4+D tensor with shape: `batch_shape + (channels, rows, cols)` if
        `data_format='channels_first'`
        or 4+D tensor with shape: `batch_shape + (rows, cols, channels)` if
        `data_format='channels_last'`.
    Output shape:
        4+D tensor with shape: `batch_shape + (filters, new_rows, new_cols)` if
        `data_format='channels_first'` or 4+D tensor with shape: `batch_shape +
        (new_rows, new_cols, filters)` if `data_format='channels_last'`.  `rows`
        and `cols` values might have changed due to padding.
    Returns:
        A tensor of rank 4+ representing
        `activation(adder2d(inputs, kernel) + bias)`.
    Raises:
        ValueError: if `padding` is `"causal"`.
        ValueError: when both `strides > 1` and `dilation_rate > 1`.
    """

    def __init__(self,
                 #  rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 groups=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',  # random_normal
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 conv_op=None,
                 **kwargs):
        super(Adder2D, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        rank = 2
        self.rank = rank

        if isinstance(filters, float):
            filters = int(filters)
        self.filters = filters
        self.groups = groups or 1
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=self.rank + 2)

        self._validate_init()
        self._is_causal = self.padding == 'causal'
        self._channels_first = self.data_format == 'channels_first'
        self._tf_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2)

    def _validate_init(self):
        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(
                'The number of filters must be evenly divisible by the number of '
                'groups. Received: groups={}, filters={}'.format(
                    self.groups, self.filters))

        if not all(self.kernel_size):
            raise ValueError('The argument `kernel_size` cannot contain 0(s). '
                             'Received: %s' % (self.kernel_size,))

        # if expression:
        #   pass (self.padding == 'causal' and not isinstance(self,
        #                                                 (Conv1D, SeparableConv1D))):
        #     raise ValueError('Causal padding is only supported for `Conv1D`'
        #                      'and `SeparableConv1D`.')

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)

        # kernel_shape: (h_filter, w_filter, in_channels, n_filters/out_channels)
        # print(self.kernel_size)
        kernel_shape = self.kernel_size + (input_channel, self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        # (h_filter, w_filter, in_channels, n_filters/out_channels)
        h_filter, w_filter, d_filter, n_filters = self.kernel.shape
        n_x, h_x, w_x, d_x = inputs.shape
        n_x = tf.shape(inputs)[0]  # actual shape

        # extract_patches takes a 4-D Tensor with shape [batch, in_rows, in_cols, depth] as input
        patches = tf.image.extract_patches(
            # .upper()
            inputs, sizes=[1, h_filter, w_filter, 1], strides=[1, self.strides[0], self.strides[1], 1], rates=[1, 1, 1, 1], padding=self.padding.upper()
        )

        n_out, h_out, w_out, d_out = patches.shape

        # reshape X_col and W_col for conv
        X_col = tf.reshape(patches, [-1, patches.shape[-1]])
        W_col = tf.reshape(self.kernel, [-1, n_filters])  # n_filters last

        # adder conv
        outputs = tf.abs((tf.expand_dims(W_col, 0)-tf.expand_dims(X_col, 2)))
        outputs = - tf.reduce_sum(outputs, 1)

        # Optional, custom gradient
        # outputs = adder2d_im2col(X_col, W_col)

        # reshape outputs back
        # n_filters index last
        outputs = tf.reshape(outputs, [n_x, h_out, w_out, n_filters])

        if self.use_bias:
            outputs += self.bias[tf.newaxis, tf.newaxis, tf.newaxis, :]

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return -1 - self.rank
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        return int(input_shape[channel_axis])

    def get_config(self):
        config = {
            'filters':
                self.filters,
            'kernel_size':
                self.kernel_size,
            'strides':
                self.strides,
            'padding':
                self.padding,
            'data_format':
                self.data_format,
            'dilation_rate':
                self.dilation_rate,
            'groups':
                self.groups,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint)
        }
        base_config = super(Adder2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
