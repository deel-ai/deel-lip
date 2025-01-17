# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains extra activation functions which respect the Lipschitz constant.
It can be added as a layer, or it can be used in the "activation" params for other
layers.
"""
import math
import keras.ops as K
from keras.constraints import MinMaxNorm
from keras.layers import Layer, PReLU
from .base_layer import LipschitzLayer
from keras.saving import register_keras_serializable


@register_keras_serializable("deel-lip", "MaxMin")
class MaxMin(Layer, LipschitzLayer):
    def __init__(self, data_format="channels_last", k_coef_lip=1.0, **kwargs):
        """
        MaxMin activation [Relu(x),reLU(-x)]

        Args:
            data_format (str): either channels_first or channels_last
            k_coef_lip (float): the lipschitz coefficient to be enforced
            **kwargs: params passed to layers (named fashion)

        Input shape:
            Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does
            not include the samples axis) when using this layer as the first layer in a
            model.

        Output shape:
            Double channel size as input.

        References:
            ([M. Blot, M. Cord, et N. Thome, « Max-min convolutional neural networks
            for image classification », in 2016 IEEE International Conference on Image
            Processing (ICIP), Phoenix, AZ, USA, 2016, p. 3678‑3682.)

        """
        self.set_klip_factor(k_coef_lip)
        super(MaxMin, self).__init__(**kwargs)
        if data_format == "channels_last":
            self.channel_axis = -1
        elif data_format == "channels_first":
            self.channel_axis = 1
        else:
            raise RuntimeError("data format not understood")
        self.data_format = data_format

    def build(self, input_shape):
        self._init_lip_coef(input_shape)
        return super().build(input_shape)

    def _compute_lip_coef(self, input_shape=None):
        return 1.0

    def call(self, x):
        return (
            K.concatenate(
                (K.relu(x, alpha=0), K.relu(-x, alpha=0)), axis=self.channel_axis
            )
            * self._get_coef()
        )

    def get_config(self):
        config = {
            "data_format": self.data_format,
            "k_coef_lip": self.k_coef_lip,
        }
        base_config = super(MaxMin, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        new_shape = input_shape
        new_shape[self.channel_axis] = 2 * new_shape[self.channel_axis]
        return new_shape


@register_keras_serializable("deel-lip", "GroupSort")
class GroupSort(Layer, LipschitzLayer):
    def __init__(self, n=None, data_format="channels_last", k_coef_lip=1.0, **kwargs):
        """
        GroupSort activation

        Args:
            n (int): group size used when sorting. When None group size is set to input
                size (fullSort behavior)
            data_format (str): either channels_first or channels_last
            k_coef_lip (float): the lipschitz coefficient to be enforced
            **kwargs: params passed to layers (named fashion)

        Input shape:
            Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does
            not include the samples axis) when using this layer as the first layer in a
            model.

        Output shape:
            Same size as input.

        """
        self.set_klip_factor(k_coef_lip)
        super(GroupSort, self).__init__(**kwargs)
        if data_format == "channels_last":
            self.channel_axis = -1
        elif data_format == "channels_first":
            raise RuntimeError(
                "channels_first not implemented for GroupSort activation"
            )
        else:
            raise RuntimeError("data format not understood")
        self.n = n
        self.data_format = data_format

    def build(self, input_shape):
        super(GroupSort, self).build(input_shape)
        self._init_lip_coef(input_shape)
        if (self.n is None) or (self.n > input_shape[self.channel_axis]):
            self.n = input_shape[self.channel_axis]
        if (input_shape[self.channel_axis] % self.n) != 0:
            raise RuntimeError("self.n has to be a divisor of the number of channels")
        input_shape = tuple(input_shape)
        self.flat_shape = (
            (-1,) + input_shape[1:-1] + (input_shape[-1] // self.n, self.n)
        )
        self.out_shape = (-1,) + input_shape[1:]

    def _compute_lip_coef(self, input_shape=None):
        return 1.0

    def call(self, x):
        fv = K.reshape(x, self.flat_shape)
        if self.n == 2:
            b, c = K.split(fv, 2, axis=-1)
            newv = K.concatenate([K.minimum(b, c), K.maximum(b, c)], axis=-1)
            newv = K.reshape(newv, self.out_shape)
            return newv * self._get_coef()

        newv = K.sort(fv)
        newv = K.reshape(newv, self.out_shape)
        return newv * self._get_coef()

    def get_config(self):
        config = {
            "n": self.n,
            "k_coef_lip": self.k_coef_lip,
            "data_format": self.data_format,
        }
        base_config = super(GroupSort, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


@register_keras_serializable("deel-lip", "GroupSort2")
class GroupSort2(GroupSort):
    def __init__(self, **kwargs):
        """
        GroupSort2 activation. Special case of GroupSort with group of size 2.

        Input shape:
            Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does
            not include the samples axis) when using this layer as the first layer in a
            model.

        Output shape:
            Same size as input.

        """
        kwargs["n"] = 2
        super().__init__(**kwargs)


@register_keras_serializable("deel-lip", "FullSort")
class FullSort(GroupSort):
    def __init__(self, **kwargs):
        """
        FullSort activation. Special case of GroupSort where the entire input is sorted.

        Input shape:
            Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does
            not include the samples axis) when using this layer as the first layer in a
            model.

        Output shape:
            Same size as input.

        """
        kwargs["n"] = None
        super().__init__(**kwargs)


@register_keras_serializable("deel-lip", "PReLUlip")
def PReLUlip(k_coef_lip=1.0):
    """
    PreLu activation, with Lipschitz constraint.

    Args:
        k_coef_lip (float): lipschitz coefficient to be enforced
    """
    return PReLU(
        alpha_constraint=MinMaxNorm(min_value=-k_coef_lip, max_value=k_coef_lip)
    )


@register_keras_serializable("deel-lip", "Householder")
class Householder(Layer, LipschitzLayer):
    def __init__(
        self,
        data_format="channels_last",
        k_coef_lip=1.0,
        theta_initializer=None,
        **kwargs,
    ):
        """
        Householder activation:
        [this review](https://openreview.net/pdf?id=tD7eCtaSkR)
        From [this repository](https://github.com/singlasahil14/SOC)

        Args:
            data_format (str): either channels_first or channels_last. Only
                channels_last is supported.
            k_coef_lip (str): The lipschitz coefficient to be enforced.
            theta_initializer: initializer for the angle theta of reflection. Defaults
                to pi/2, which corresponds to GroupSort2.
            **kwargs: parameters passed to the `keras.layers.Layer`.

        Input shape:
            Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does
            not include the samples axis) when using this layer as the first layer in a
            model.

        Output shape:
            Same size as input.

        """
        if data_format != "channels_last":
            raise RuntimeError("Only 'channels_last' data format is supported")

        self.data_format = data_format
        self.set_klip_factor(k_coef_lip)
        self.theta_initializer = theta_initializer
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self._init_lip_coef(input_shape)
        if (input_shape[-1] % 2) != 0:
            raise RuntimeError("2 has to be a divisor of the number of channels")

        self.theta = self.add_weight(
            name="theta",
            shape=[input_shape[-1] // 2],
            initializer=self.theta_initializer,
        )
        if self.theta_initializer is None:
            self.theta.assign(K.ones_like(self.theta, dtype="float32") * math.pi / 2)

    def _compute_lip_coef(self, input_shape=None):
        return 1.0

    def call(self, x):
        z1, z2 = K.split(x, 2, axis=-1)

        # selector > 0 if point (z1, z2) is on one side of reflection line, else < 0.
        # Reflection line is defined by angle theta/2.
        selector = (z1 * K.sin(0.5 * self.theta)) - (z2 * K.cos(0.5 * self.theta))

        cos_theta = K.cos(self.theta)
        sin_theta = K.sin(self.theta)
        reflected_z1 = z1 * cos_theta + z2 * sin_theta
        reflected_z2 = z1 * sin_theta - z2 * cos_theta

        a = K.where(selector <= 0, z1, reflected_z1)
        b = K.where(selector <= 0, z2, reflected_z2)

        return K.concatenate([a, b], axis=-1)

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "data_format": self.data_format,
            "theta_initializer": self.theta_initializer,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
