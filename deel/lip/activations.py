# © IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All rights reserved. DEEL is a research
# program operated by IVADO, IRT Saint Exupéry, CRIAQ and ANITI - https://www.deel.ai/
"""
This module contains extra activation functions which respect the Lipschitz constant. It can be added as a layer,
or it can be used in the "activation" params for other layers.
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.layers import Layer, PReLU
from .layers import LipschitzLayer
from .utils import _deel_export


@_deel_export
class MaxMin(Layer, LipschitzLayer):
    def __init__(self, data_format="channels_last", k_coef_lip=1.0, *args, **kwargs):
        """
        MaxMin activation [Relu(x),reLU(-x)]

        Args:
            data_format: either channels_first or channels_last
            k_coef_lip: the lipschitz coefficient to be enforced
            *args: params passed to Layers
            **kwargs: params passed to layers (named fashion)

        Input shape:
            Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis)
            when using this layer as the first layer in a model.

        Output shape:
            Double channel size as input.

        References:
            ([M. Blot, M. Cord, et N. Thome, « Max-min convolutional neural networks for image classification »,
            in 2016 IEEE International Conference on Image Processing (ICIP), Phoenix, AZ, USA, 2016, p. 3678‑3682.)

        """
        self.set_klip_factor(k_coef_lip)
        super(MaxMin, self).__init__(*args, **kwargs)
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

    def call(self, x, **kwargs):
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


@_deel_export
class GroupSort(Layer, LipschitzLayer):
    def __init__(
        self, n=None, data_format="channels_last", k_coef_lip=1.0, *args, **kwargs
    ):
        """
        GroupSort activation

        Args:
            n: group size used when sorting. When None group size is set to input size (fullSort behavior)
            data_format: either channels_first or channels_last
            k_coef_lip: the lipschitz coefficient to be enforced
            *args: params passed to Layers
            **kwargs: params passed to layers (named fashion)

        Input shape:
            Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis)
            when using this layer as the first layer in a model.

        Output shape:
            Same size as input.

        """
        self.set_klip_factor(k_coef_lip)
        super(GroupSort, self).__init__(*args, **kwargs)
        if data_format == "channels_last":
            self.channel_axis = -1
        elif data_format == "channels_first":
            raise RuntimeError(
                "channels_first not implemented for GroupSort activation"
            )
            self.channel_axis = 1
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
        print(self.n)

    def _compute_lip_coef(self, input_shape=None):
        return 1.0

    def call(self, x, **kwargs):
        fv = tf.reshape(x, [-1, self.n])
        if self.n == 2:
            b, c = tf.split(fv, 2, 1)
            newv = tf.concat([tf.minimum(b, c), tf.maximum(b, c)], axis=1)
            newv = tf.reshape(newv, tf.shape(x))
            return newv

        newv = tf.sort(fv)
        newv = tf.reshape(newv, tf.shape(x))
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


@_deel_export
class GroupSort2(GroupSort):

    def __init__(self, **kwargs):
        """
        GroupSort2 activation. Special case of GroupSort with group of size 2.

        Input shape:
            Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis)
            when using this layer as the first layer in a model.

        Output shape:
            Same size as input.

        """
        kwargs["n"] = 2
        super().__init__(**kwargs)


@_deel_export
class FullSort(GroupSort):

    def __init__(self, **kwargs):
        """
        FullSort activation. Special case of GroupSort where the entire input is sorted.

        Input shape:
            Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis)
            when using this layer as the first layer in a model.

        Output shape:
            Same size as input.

        """
        kwargs["n"] = None
        super().__init__(**kwargs)


@_deel_export
def PReLUlip(k_coef_lip=1.0):
    """
    PreLu activation, with Lipschitz constraint.

    Args:
        k_coef_lip: lipschitz coefficient to be enforced
    """
    return PReLU(
        alpha_constraint=MinMaxNorm(min_value=-k_coef_lip, max_value=k_coef_lip)
    )
