# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module extends original keras layers, in order to add k lipschitz constraint via
reparametrization. Currently, are implemented:
* Dense layer:
    as SpectralDense (and as FrobeniusDense when the layer has a single
    output)
* Conv2D layer:
    as SpectralConv2D (and as FrobeniusConv2D when the layer has a single
    output)
* AveragePooling:
    as ScaledAveragePooling
* GlobalAveragePooling2D:
    as ScaledGlobalAveragePooling2D
By default the layers are 1 Lipschitz almost everywhere, which is efficient for
wasserstein distance estimation. However for other problems (such as adversarial
robustness) the user may want to use layers that are at most 1 lipschitz, this can
be done by setting the param `eps_bjorck=None`.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as keraslayers
from tensorflow.keras.utils import register_keras_serializable

from .base_layer import LipschitzLayer


@register_keras_serializable("deel-lip", "ScaledAveragePooling2D")
class ScaledAveragePooling2D(keraslayers.AveragePooling2D, LipschitzLayer):
    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        k_coef_lip=1.0,
        **kwargs,
    ):
        """
        Average pooling operation for spatial data, but with a lipschitz bound.

        Arguments:
            pool_size: integer or tuple of 2 integers,
                factors by which to downscale (vertical, horizontal).
                `(2, 2)` will halve the input in both spatial dimension.
                If only one integer is specified, the same window length
                will be used for both dimensions.
            strides: Integer, tuple of 2 integers, or None.
                Strides values.
                If None, it will default to `pool_size`.
            padding: One of `"valid"` or `"same"` (case-insensitive).
            data_format: A string,
                one of `channels_last` (default) or `channels_first`.
                The ordering of the dimensions in the inputs.
                `channels_last` corresponds to inputs with shape
                `(batch, height, width, channels)` while `channels_first`
                corresponds to inputs with shape
                `(batch, channels, height, width)`.
                It defaults to the `image_data_format` value found in your
                Keras config file at `~/.keras/keras.json`.
                If you never set it, then it will be "channels_last".
            k_coef_lip: the lipschitz factor to ensure

        Input shape:
            - If `data_format='channels_last'`:
                4D tensor with shape `(batch_size, rows, cols, channels)`.
            - If `data_format='channels_first'`:
                4D tensor with shape `(batch_size, channels, rows, cols)`.

        Output shape:
            - If `data_format='channels_last'`:
                4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
            - If `data_format='channels_first'`:
                4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.

        This documentation reuse the body of the original keras.layers.AveragePooling2D
        doc.
        """
        if not ((strides == pool_size) or (strides is None)):
            raise RuntimeError("stride must be equal to pool_size")
        if padding != "valid":
            raise RuntimeError("ScaledAveragePooling2D only supports padding='valid'")
        super(ScaledAveragePooling2D, self).__init__(
            pool_size=pool_size,
            strides=pool_size,
            padding=padding,
            data_format=data_format,
            **kwargs,
        )
        self.set_klip_factor(k_coef_lip)
        self._kwargs = kwargs

    def build(self, input_shape):
        super(ScaledAveragePooling2D, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        return np.sqrt(np.prod(np.asarray(self.pool_size)))

    def call(self, x):
        return super(keraslayers.AveragePooling2D, self).call(x) * self._get_coef()

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
        }
        base_config = super(ScaledAveragePooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "ScaledL2NormPooling2D")
class ScaledL2NormPooling2D(keraslayers.AveragePooling2D, LipschitzLayer):
    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        k_coef_lip=1.0,
        eps_grad_sqrt=1e-6,
        **kwargs,
    ):
        """
        Average pooling operation for spatial data, with a lipschitz bound. This
        pooling operation is norm preserving (aka gradient=1 almost everywhere).

        [1]Y.-L.Boureau, J.Ponce, et Y.LeCun, « A Theoretical Analysis of Feature
        Pooling in Visual Recognition »,p.8.

        Arguments:
            pool_size: integer or tuple of 2 integers,
                factors by which to downscale (vertical, horizontal).
                `(2, 2)` will halve the input in both spatial dimension.
                If only one integer is specified, the same window length
                will be used for both dimensions.
            strides: Integer, tuple of 2 integers, or None.
                Strides values.
                If None, it will default to `pool_size`.
            padding: One of `"valid"` or `"same"` (case-insensitive).
            data_format: A string,
                one of `channels_last` (default) or `channels_first`.
                The ordering of the dimensions in the inputs.
                `channels_last` corresponds to inputs with shape
                `(batch, height, width, channels)` while `channels_first`
                corresponds to inputs with shape
                `(batch, channels, height, width)`.
                It defaults to the `image_data_format` value found in your
                Keras config file at `~/.keras/keras.json`.
                If you never set it, then it will be "channels_last".
            k_coef_lip: the lipschitz factor to ensure
            eps_grad_sqrt: Epsilon value to avoid numerical instability
                due to non-defined gradient at 0 in the sqrt function

        Input shape:
            - If `data_format='channels_last'`:
                4D tensor with shape `(batch_size, rows, cols, channels)`.
            - If `data_format='channels_first'`:
                4D tensor with shape `(batch_size, channels, rows, cols)`.

        Output shape:
            - If `data_format='channels_last'`:
                4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
            - If `data_format='channels_first'`:
                4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.
        """
        if not ((strides == pool_size) or (strides is None)):
            raise RuntimeError("stride must be equal to pool_size")
        if padding != "valid":
            raise RuntimeError("ScaledL2NormPooling2D only supports padding='valid'")
        if eps_grad_sqrt < 0.0:
            raise RuntimeError("eps_grad_sqrt must be positive")
        super(ScaledL2NormPooling2D, self).__init__(
            pool_size=pool_size,
            strides=pool_size,
            padding=padding,
            data_format=data_format,
            **kwargs,
        )
        self.set_klip_factor(k_coef_lip)
        self.eps_grad_sqrt = eps_grad_sqrt
        self._kwargs = kwargs

    def build(self, input_shape):
        super(ScaledL2NormPooling2D, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        return np.sqrt(np.prod(np.asarray(self.pool_size)))

    @staticmethod
    def _sqrt(eps_grad_sqrt):
        @tf.custom_gradient
        def sqrt_op(x):
            sqrtx = tf.sqrt(x)

            def grad(dy):
                return dy / (2 * (sqrtx + eps_grad_sqrt))

            return sqrtx, grad

        return sqrt_op

    def call(self, x):
        return (
            ScaledL2NormPooling2D._sqrt(self.eps_grad_sqrt)(
                super(ScaledL2NormPooling2D, self).call(tf.square(x))
            )
            * self._get_coef()
        )

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
        }
        base_config = super(ScaledL2NormPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "ScaledGlobalL2NormPooling2D")
class ScaledGlobalL2NormPooling2D(keraslayers.GlobalAveragePooling2D, LipschitzLayer):
    def __init__(self, data_format=None, k_coef_lip=1.0, eps_grad_sqrt=1e-6, **kwargs):
        """
        Average pooling operation for spatial data, with a lipschitz bound. This
        pooling operation is norm preserving (aka gradient=1 almost everywhere).

        [1]Y.-L.Boureau, J.Ponce, et Y.LeCun, « A Theoretical Analysis of Feature
        Pooling in Visual Recognition »,p.8.

        Arguments:
            data_format: A string,
                one of `channels_last` (default) or `channels_first`.
                The ordering of the dimensions in the inputs.
                `channels_last` corresponds to inputs with shape
                `(batch, height, width, channels)` while `channels_first`
                corresponds to inputs with shape
                `(batch, channels, height, width)`.
                It defaults to the `image_data_format` value found in your
                Keras config file at `~/.keras/keras.json`.
                If you never set it, then it will be "channels_last".
            k_coef_lip: the lipschitz factor to ensure
            eps_grad_sqrt: Epsilon value to avoid numerical instability
                due to non-defined gradient at 0 in the sqrt function

        Input shape:
            - If `data_format='channels_last'`:
                4D tensor with shape `(batch_size, rows, cols, channels)`.
            - If `data_format='channels_first'`:
                4D tensor with shape `(batch_size, channels, rows, cols)`.

        Output shape:
            - If `data_format='channels_last'`:
                4D tensor with shape `(batch_size, channels)`.
            - If `data_format='channels_first'`:
                4D tensor with shape `(batch_size, pooled_cols)`.
        """
        if eps_grad_sqrt < 0.0:
            raise RuntimeError("eps_grad_sqrt must be positive")
        super(ScaledGlobalL2NormPooling2D, self).__init__(
            data_format=data_format, **kwargs
        )
        self.set_klip_factor(k_coef_lip)
        self.eps_grad_sqrt = eps_grad_sqrt
        self._kwargs = kwargs
        if self.data_format == "channels_last":
            self.axes = [1, 2]
        else:
            self.axes = [2, 3]

    def build(self, input_shape):
        super(ScaledGlobalL2NormPooling2D, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        return 1.0

    @staticmethod
    def _sqrt(eps_grad_sqrt):
        @tf.custom_gradient
        def sqrt_op(x):
            sqrtx = tf.sqrt(x)

            def grad(dy):
                return dy / (2 * (sqrtx + eps_grad_sqrt))

            return sqrtx, grad

        return sqrt_op

    def call(self, x):
        return (
            ScaledL2NormPooling2D._sqrt(self.eps_grad_sqrt)(
                tf.reduce_sum(tf.square(x), axis=self.axes)
            )
            * self._get_coef()
        )

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
        }
        base_config = super(ScaledGlobalL2NormPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "ScaledGlobalAveragePooling2D")
class ScaledGlobalAveragePooling2D(keraslayers.GlobalAveragePooling2D, LipschitzLayer):
    def __init__(self, data_format=None, k_coef_lip=1.0, **kwargs):
        """Global average pooling operation for spatial data with Lipschitz bound.

        Arguments:
            data_format: A string,
                one of `channels_last` (default) or `channels_first`.
                The ordering of the dimensions in the inputs.
                `channels_last` corresponds to inputs with shape
                `(batch, height, width, channels)` while `channels_first`
                corresponds to inputs with shape
                `(batch, channels, height, width)`.
                It defaults to the `image_data_format` value found in your
                Keras config file at `~/.keras/keras.json`.
                If you never set it, then it will be "channels_last".

        Input shape:
            - If `data_format='channels_last'`:
                4D tensor with shape `(batch_size, rows, cols, channels)`.
            - If `data_format='channels_first'`:
                4D tensor with shape `(batch_size, channels, rows, cols)`.

        Output shape:
        2D tensor with shape `(batch_size, channels)`.

        This documentation reuse the body of the original
        keras.layers.GlobalAveragePooling doc.
        """
        super(ScaledGlobalAveragePooling2D, self).__init__(
            data_format=data_format, **kwargs
        )
        self.set_klip_factor(k_coef_lip)
        self._kwargs = kwargs

    def build(self, input_shape):
        super(ScaledGlobalAveragePooling2D, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        if self.data_format == "channels_last":
            lip_coef = np.sqrt(input_shape[-3] * input_shape[-2])
        elif self.data_format == "channels_first":
            lip_coef = np.sqrt(input_shape[-2] * input_shape[-1])
        else:
            raise RuntimeError(f"data format not understood: {self.data_format}")
        return lip_coef

    def call(self, x):
        return super(ScaledGlobalAveragePooling2D, self).call(x) * self._get_coef()

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
        }
        base_config = super(ScaledGlobalAveragePooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "InvertibleDownSampling")
class InvertibleDownSampling(keraslayers.Layer):
    def __init__(
        self, pool_size, data_format="channels_last", name=None, dtype=None, **kwargs
    ):
        """

        This pooling layer perform a reshape on the spacial dimensions: it take a
        (bs, h, w, c) ( if channels_last ) and reshape it to a
        (bs, h/p_h, w/p_w, c*p_w*p_h ), where p_w and p_h are the shape of the pool.
        By doing this the image size is reduced while the number of channels is
        increased.

        References:
            Anil et al. [paper](https://arxiv.org/abs/1911.00937)

        Note:
            The image shape must be divisible by the pool shape.

        Args:
            pool_size: tuple describing the pool shape
            data_format: can either be `channels_last` or `channels_first`
            name: name of the layer
            dtype: dtype of the layer
            **kwargs: params passed to the Layers constructor
        """
        super(InvertibleDownSampling, self).__init__(name=name, dtype=dtype, **kwargs)
        self.pool_size = pool_size
        self.data_format = data_format

    def call(self, inputs):
        if self.data_format == "channels_last":
            return tf.concat(
                [
                    inputs[
                        :, i :: self.pool_size[0], j :: self.pool_size[1], :
                    ]  # for now we handle only channels last
                    for i in range(self.pool_size[0])
                    for j in range(self.pool_size[1])
                ],
                axis=-1,
            )
        else:
            return tf.concat(
                [
                    inputs[
                        :, :, i :: self.pool_size[0], j :: self.pool_size[1]
                    ]  # for now we handle only channels last
                    for i in range(self.pool_size[0])
                    for j in range(self.pool_size[1])
                ],
                axis=1,
            )

    def get_config(self):
        config = {
            "data_format": self.data_format,
            "pool_size": self.pool_size,
        }
        base_config = super(InvertibleDownSampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "InvertibleUpSampling")
class InvertibleUpSampling(keraslayers.Layer):
    def __init__(
        self, pool_size, data_format="channels_last", name=None, dtype=None, **kwargs
    ):
        """

        This Layer is the inverse of the InvertibleDownSampling layer. It take a
        (bs, h, w, c) ( if channels_last ) and reshape it to a
        (bs, h/p_h, w/p_w, c*p_w*p_h ), where p_w and p_h are the shape of the
        pool. By doing this the image size is reduced while the number of
        channels is increased.

        References:
            Anil et al. [paper](https://arxiv.org/abs/1911.00937)

        Note:
            The input number of channels must be divisible by the `p_w*p_h`.


        Args:
            pool_size: tuple describing the pool shape (p_h, p_w)
            data_format: can either be `channels_last` or `channels_first`
            name: name of the layer
            dtype: dtype of the layer
            **kwargs: params passed to the Layers constructor
        """
        super(InvertibleUpSampling, self).__init__(name=name, dtype=dtype, **kwargs)
        self.pool_size = pool_size
        self.data_format = data_format

    def call(self, inputs):
        if self.data_format == "channels_first":
            # convert to channels_first
            inputs = tf.transpose(inputs, [0, 2, 3, 1])
        # from shape (bs, w, h, c*pw*ph) to (bs, w, h, pw, ph, c)
        input_shape = tf.shape(inputs)
        w, h, c_in = input_shape[1], input_shape[2], input_shape[3]
        pw, ph = self.pool_size
        c = c_in // (pw * ph)
        inputs = tf.reshape(inputs, (-1, w, h, pw, ph, c))
        inputs = tf.transpose(
            tf.reshape(
                tf.transpose(
                    inputs, [0, 5, 2, 4, 1, 3]
                ),  # (bs, w, h, pw, ph, c) -> (bs, c, w, pw, h, ph)
                (-1, c, w, pw, h * ph),
            ),  # (bs, c, w, pw, h, ph) -> (bs, c, w, pw, h*ph) merge last axes
            [
                0,
                1,
                4,
                2,
                3,
            ],  # (bs, c, w, pw, h*ph) -> (bs, c, h*ph, w, pw)
            # put each axis back in place
        )
        inputs = tf.reshape(
            inputs, (-1, c, h * ph, w * pw)
        )  # (bs, c, h*ph, w, pw) -> (bs, c, h*ph, w*pw)
        if self.data_format == "channels_last":
            inputs = tf.transpose(
                inputs, [0, 2, 3, 1]  # (bs, c, h*ph, w*pw) -> (bs, w*pw, h*ph, c)
            )
        return inputs

    def get_config(self):
        config = {
            "data_format": self.data_format,
            "pool_size": self.pool_size,
        }
        base_config = super(InvertibleUpSampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
