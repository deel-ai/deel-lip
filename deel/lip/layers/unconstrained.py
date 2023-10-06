# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module contains custom Keras unconstrained layers.

Compared to other files in `layers` folder, the layers defined here are not
Lipschitz-constrained. They are base classes for more advanced layers. Do not use these
layers as is, since they are not Lipschitz constrained.
"""
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

from ..utils import _padding_circular
from .base_layer import Condensable


@register_keras_serializable("deel-lip", "PadConv2D")
class PadConv2D(tf.keras.layers.Conv2D, Condensable):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        """
        This class is a Conv2D Layer with parameterized padding.
        Since Conv2D layer only supports `"same"` and `"valid"` padding, this layer will
        enable other type of padding, such as `"constant"`, `"symmetric"`, `"reflect"`
        or `"circular"`.

        Warning:
            The PadConv2D is not a Lipschitz layer and must not be directly used. This
            must be used as a base class to create a Lipschitz layer with padding.

        All arguments are the same as the original `Conv2D` except the `padding`
        which is defined as following:

        Args:
            padding: one of `"same"`, `"valid"` `"constant"`, `"symmetric"`,
                `"reflect"` or `"circular"` (case-insensitive).
        """
        self.pad = lambda x: x
        self.old_padding = padding
        self.internal_input_shape = None
        if padding.lower() != "same":  # same is directly processed in Conv2D
            padding = "valid"
        super(PadConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self._kwargs = kwargs
        if self.old_padding.lower() in ["same", "valid"]:
            self.pad = lambda x: x
            self.padding_size = [0, 0]
        if self.old_padding.lower() in ["constant", "reflect", "symmetric"]:
            self.padding_size = [self.kernel_size[0] // 2, self.kernel_size[1] // 2]
            paddings = [
                [0, 0],
                [self.padding_size[0], self.padding_size[0]],
                [self.padding_size[1], self.padding_size[1]],
                [0, 0],
            ]
            self.pad = lambda t: tf.pad(t, paddings, self.old_padding)
        if self.old_padding.lower() == "circular":
            self.padding_size = [self.kernel_size[0] // 2, self.kernel_size[1] // 2]
            self.pad = lambda t: _padding_circular(t, self.padding_size)

    def _compute_padded_shape(self, input_shape, padding_size):
        if isinstance(input_shape, tf.TensorShape):
            internal_input_shape = input_shape.as_list()
        else:
            internal_input_shape = list(input_shape)

        first_spatial_dim = 1 if self.data_format == "channels_last" else 2
        for index, pad in enumerate(padding_size):
            internal_input_shape[first_spatial_dim + index] += 2 * pad
        return tf.TensorShape(internal_input_shape)

    def build(self, input_shape):
        self.internal_input_shape = self._compute_padded_shape(
            input_shape, self.padding_size
        )
        super(PadConv2D, self).build(self.internal_input_shape)

    def compute_output_shape(self, input_shape):
        return super(PadConv2D, self).compute_output_shape(self.internal_input_shape)

    def call(self, x):
        x = self.pad(x)
        return super(PadConv2D, self).call(x)

    def get_config(self):
        base_config = super(PadConv2D, self).get_config()
        base_config["padding"] = self.old_padding
        return base_config

    def condense(self):
        return

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        if self.old_padding.lower() in ["same", "valid"]:
            layer_type = tf.keras.layers.Conv2D
        else:
            layer_type = PadConv2D
        layer = layer_type(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.old_padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            **self._kwargs
        )
        layer.build(self.input_shape)
        layer.kernel.assign(self.kernel)
        if self.use_bias:
            layer.bias.assign(self.bias)
        return layer
