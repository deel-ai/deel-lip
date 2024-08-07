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
import keras
import keras.ops as K
from keras.initializers import RandomNormal
from keras.layers import Conv2D, Conv2DTranspose
from keras.saving import register_keras_serializable

from ..constraints import SpectralConstraint
from ..initializers import SpectralInitializer
from ..normalizers import (
    DEFAULT_BETA_BJORCK,
    DEFAULT_EPS_BJORCK,
    DEFAULT_EPS_SPECTRAL,
    DEFAULT_MAXITER_BJORCK,
    DEFAULT_MAXITER_SPECTRAL,
    _check_RKO_params,
    reshaped_kernel_orthogonalization,
)
from .base_layer import Condensable, LipschitzLayer


def _compute_conv_lip_factor(kernel_size, strides, input_shape, data_format):
    """Compute the Lipschitz factor to apply on estimated Lipschitz constant in
    convolutional layer. This factor depends on the kernel size, the strides and the
    input shape.
    """
    stride = np.prod(strides)
    kh, kw = kernel_size[0], kernel_size[1]
    kh_div2 = (kh - 1) / 2
    kw_div2 = (kw - 1) / 2

    if data_format == "channels_last":
        h, w = input_shape[-3], input_shape[-2]
    elif data_format == "channels_first":
        h, w = input_shape[-2], input_shape[-1]
    else:
        raise RuntimeError(f"data_format not understood: {data_format}")

    if stride == 1:
        return np.sqrt(
            (w * h)
            / ((kh * h - kh_div2 * (kh_div2 + 1)) * (kw * w - kw_div2 * (kw_div2 + 1)))
        )
    else:
        return np.sqrt(1.0 / (np.ceil(kh / strides[0]) * np.ceil(kw / strides[1])))


@register_keras_serializable("deel-lip", "SpectralConv2D")
class SpectralConv2D(Conv2D, LipschitzLayer, Condensable):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer=SpectralInitializer(),
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        k_coef_lip=1.0,
        eps_spectral=DEFAULT_EPS_SPECTRAL,
        eps_bjorck=DEFAULT_EPS_BJORCK,
        beta_bjorck=DEFAULT_BETA_BJORCK,
        maxiter_spectral=DEFAULT_MAXITER_SPECTRAL,
        maxiter_bjorck=DEFAULT_MAXITER_BJORCK,
        **kwargs,
    ):
        """
        This class is a Conv2D Layer constrained such that all singular of it's kernel
        are 1. The computation based on Bjorck algorithm. As this is not
        enough to ensure 1 Lipschitzity a coertive coefficient is applied on the
        output.
        The computation is done in three steps:

        1. reduce the largest singular value to 1, using iterated power method.
        2. increase other singular values to 1, using Bjorck algorithm.
        3. divide the output by the Lipschitz bound to ensure k Lipschitzity.

        Args:
            filters: Integer, the dimensionality of the output space
                (i.e. the number of output filters in the convolution).
            kernel_size: An integer or tuple/list of 2 integers, specifying the
                height and width of the 2D convolution window.
                Can be a single integer to specify the same value for
                all spatial dimensions.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the height and width.
                Can be a single integer to specify the same value for
                all spatial dimensions.
                Specifying any stride value != 1 is incompatible with specifying
                any `dilation_rate` value != 1.
            padding: one of `"valid"` or `"same"` (case-insensitive).
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
            dilation_rate: an integer or tuple/list of 2 integers, specifying
                the dilation rate to use for dilated convolution.
                Can be a single integer to specify the same value for
                all spatial dimensions.
                Currently, specifying any `dilation_rate` value != 1 is
                incompatible with specifying any stride value != 1.
            groups: A positive int specifying the number of groups in which the input is
                split along the channel axis. This layer only supports groups=1.
            activation: Activation function to use.
                If you don't specify anything, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Regularizer function applied to
                the `kernel` weights matrix.
            bias_regularizer: Regularizer function applied to the bias vector.
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation")..
            kernel_constraint: Constraint function applied to the kernel matrix.
            bias_constraint: Constraint function applied to the bias vector.
            k_coef_lip: lipschitz constant to ensure
            eps_spectral: stopping criterion for the iterative power algorithm.
            eps_bjorck: stopping criterion Bjorck algorithm.
            beta_bjorck: beta parameter in bjorck algorithm.
            maxiter_spectral: maximum number of iterations for the power iteration.
            maxiter_bjorck: maximum number of iterations for bjorck algorithm.

        This documentation reuse the body of the original keras.layers.Conv2D doc.
        """
        if dilation_rate not in ((1, 1), [1, 1], 1):
            raise RuntimeError("SpectralConv2D does not support dilation rate")
        if padding != "same":
            raise RuntimeError("SpectralConv2D only supports padding='same'")
        if groups != 1:
            raise ValueError("SpectralConv2D does not support groups != 1")
        super(SpectralConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=1,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self._kwargs = kwargs
        self.set_klip_factor(k_coef_lip)
        self.u = None
        self.sig = None
        self.wbar = None
        _check_RKO_params(eps_spectral, eps_bjorck, beta_bjorck)
        self.eps_spectral = eps_spectral
        self.eps_bjorck = eps_bjorck
        self.beta_bjorck = beta_bjorck
        self.maxiter_bjorck = maxiter_bjorck
        self.maxiter_spectral = maxiter_spectral

    def build(self, input_shape):
        super(SpectralConv2D, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.u = self.add_weight(
            shape=(1, self.filters),
            initializer=RandomNormal(0, 1),
            name="sn",
            trainable=False,
            dtype=self.dtype,
        )

        self.sig = self.add_weight(
            shape=(1, 1),  # maximum spectral value
            name="sigma",
            trainable=False,
            dtype=self.dtype,
        )
        self.sig.assign([[1.0]])
        self.wbar = keras.Variable(self.kernel.value, trainable=False, name="wbar")
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        return _compute_conv_lip_factor(
            self.kernel_size, self.strides, input_shape, self.data_format
        )

    def call(self, x, training=True):
        if training:
            wbar, u, sigma = reshaped_kernel_orthogonalization(
                self.kernel,
                self.u,
                self._get_coef(),
                self.eps_spectral,
                self.eps_bjorck,
                self.beta_bjorck,
                self.maxiter_spectral,
                self.maxiter_bjorck,
            )
            self.wbar.assign(wbar)
            self.u.assign(u)
            self.sig.assign(sigma)
        else:
            wbar = self.wbar

        # Compute Conv2D operation (copied from keras.layers.Conv2D)
        outputs = K.conv(
            x,
            wbar,
            strides=list(self.strides),
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

        if self.use_bias:
            if self.data_format == "channels_last":
                bias_shape = (1,) * (self.rank + 1) + (self.filters,)
            else:
                bias_shape = (1, self.filters) + (1,) * self.rank
            bias = K.reshape(self.bias, bias_shape)
            outputs += bias

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "eps_spectral": self.eps_spectral,
            "eps_bjorck": self.eps_bjorck,
            "beta_bjorck": self.beta_bjorck,
            "maxiter_spectral": self.maxiter_spectral,
            "maxiter_bjorck": self.maxiter_bjorck,
        }
        base_config = super(SpectralConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def condense(self):
        wbar, u, sigma = reshaped_kernel_orthogonalization(
            self.kernel,
            self.u,
            self._get_coef(),
            self.eps_spectral,
            self.eps_bjorck,
            self.beta_bjorck,
            self.maxiter_spectral,
            self.maxiter_bjorck,
        )
        self.kernel.assign(wbar)
        self.u.assign(u)
        self.sig.assign(sigma)

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        layer = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            **self._kwargs,
        )
        layer.build(self.input.shape)
        layer.kernel.assign(self.wbar)
        if self.use_bias:
            layer.bias.assign(self.bias)
        return layer


@register_keras_serializable("deel-lip", "SpectralConv2DTranspose")
class SpectralConv2DTranspose(Conv2DTranspose, LipschitzLayer, Condensable):
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
        kernel_initializer=SpectralInitializer(),
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        k_coef_lip=1.0,
        eps_spectral=DEFAULT_EPS_SPECTRAL,
        eps_bjorck=DEFAULT_EPS_BJORCK,
        beta_bjorck=DEFAULT_BETA_BJORCK,
        maxiter_spectral=DEFAULT_MAXITER_SPECTRAL,
        maxiter_bjorck=DEFAULT_MAXITER_BJORCK,
        **kwargs,
    ):
        """
        This class is a Conv2DTranspose layer constrained such that all singular values
        of its kernel are 1. The computation is based on Björck orthogonalization
        algorithm.

        The computation is done in three steps:
        1. reduce the largest singular value to 1, using iterated power method.
        2. increase other singular values to 1, using Björck algorithm.
        3. divide the output by the Lipschitz target K to ensure K-Lipschitzity.

        This documentation reuses the body of the original
        `keras.layers.Conv2DTranspose` doc.

        Args:
            filters: Integer, the dimensionality of the output space
                (i.e. the number of output filters in the convolution).
            kernel_size: An integer or tuple/list of 2 integers, specifying the
                height and width of the 2D convolution window.
                Can be a single integer to specify the same value for
                all spatial dimensions.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the height and width.
                Can be a single integer to specify the same value for
                all spatial dimensions.
            padding: only `"same"` padding is supported in this Lipschitz layer
                (case-insensitive).
            output_padding: if set to `None` (default), the output shape is inferred.
                Only `None` value is supported in this Lipschitz layer.
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
            dilation_rate: an integer, specifying the dilation rate for all spatial
                dimensions for dilated convolution. This Lipschitz layer does not
                support dilation rate != 1.
            activation: Activation function to use.
                If you don't specify anything, no activation is applied
                (see `keras.activations`).
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix
                (see `keras.initializers`). Defaults to `SpectralInitializer`.
            bias_initializer: Initializer for the bias vector
                (see `keras.initializers`). Defaults to 'zeros'.
            kernel_regularizer: Regularizer function applied to
                the `kernel` weights matrix (see `keras.regularizers`).
            bias_regularizer: Regularizer function applied to the bias vector
                (see `keras.regularizers`).
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation") (see `keras.regularizers`).
            kernel_constraint: Constraint function applied to the kernel matrix
                (see `keras.constraints`).
            bias_constraint: Constraint function applied to the bias vector
                (see `keras.constraints`).
            k_coef_lip: Lipschitz constant to ensure
            eps_spectral: stopping criterion for the iterative power algorithm.
            eps_bjorck: stopping criterion Björck algorithm.
            beta_bjorck: beta parameter in Björck algorithm.
            maxiter_spectral: maximum number of iterations for the power iteration.
            maxiter_bjorck: maximum number of iterations for bjorck algorithm.
        """
        super().__init__(
            filters,
            kernel_size,
            strides,
            padding,
            data_format,
            dilation_rate,
            activation,
            use_bias,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            **kwargs,
        )

        if self.dilation_rate != (1, 1):
            raise ValueError("SpectralConv2DTranspose does not support dilation rate")
        if self.padding != "same":
            raise ValueError("SpectralConv2DTranspose only supports padding='same'")
        self.set_klip_factor(k_coef_lip)
        self.u = None
        self.sig = None
        self.wbar = None
        _check_RKO_params(eps_spectral, eps_bjorck, beta_bjorck)
        self.eps_spectral = eps_spectral
        self.eps_bjorck = eps_bjorck
        self.beta_bjorck = beta_bjorck
        self.maxiter_bjorck = maxiter_bjorck
        self.maxiter_spectral = maxiter_spectral
        self._kwargs = kwargs

    def build(self, input_shape):
        super().build(input_shape)
        self._init_lip_coef(input_shape)
        self.u = self.add_weight(
            shape=(1, self.filters),
            initializer=RandomNormal(0, 1),
            name="sn",
            trainable=False,
            dtype=self.dtype,
        )

        self.sig = self.add_weight(
            shape=(1, 1),  # maximum spectral value
            name="sigma",
            trainable=False,
            dtype=self.dtype,
        )
        self.sig.assign([[1.0]])
        self.wbar = keras.Variable(self.kernel.value, trainable=False, name="wbar")

    def _compute_lip_coef(self, input_shape=None):
        return _compute_conv_lip_factor(
            self.kernel_size, self.strides, input_shape, self.data_format
        )

    def call(self, inputs, training=True):
        if training:
            kernel_reshaped = K.transpose(self.kernel, [0, 1, 3, 2])
            wbar, u, sigma = reshaped_kernel_orthogonalization(
                kernel_reshaped,
                self.u,
                self._get_coef(),
                self.eps_spectral,
                self.eps_bjorck,
                self.beta_bjorck,
                self.maxiter_spectral,
                self.maxiter_bjorck,
            )
            wbar = K.transpose(wbar, [0, 1, 3, 2])
            self.wbar.assign(wbar)
            self.u.assign(u)
            self.sig.assign(sigma)
        else:
            wbar = self.wbar

        # Apply conv2D_transpose operation (copied from keras.layers.Conv2DTranspose)
        outputs = K.conv_transpose(
            inputs,
            wbar,
            strides=list(self.strides),
            padding=self.padding,
            output_padding=self.output_padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

        if self.use_bias:
            if self.data_format == "channels_last":
                bias_shape = (1,) * (self.rank + 1) + (self.filters,)
            else:
                bias_shape = (1, self.filters) + (1,) * self.rank
            bias = K.reshape(self.bias, bias_shape)
            outputs += bias

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "eps_spectral": self.eps_spectral,
            "eps_bjorck": self.eps_bjorck,
            "beta_bjorck": self.beta_bjorck,
            "maxiter_spectral": self.maxiter_spectral,
            "maxiter_bjorck": self.maxiter_bjorck,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def condense(self):
        wbar, u, sigma = reshaped_kernel_orthogonalization(
            self.kernel,
            self.u,
            self._get_coef(),
            self.eps_spectral,
            self.eps_bjorck,
            self.beta_bjorck,
            self.maxiter_spectral,
            self.maxiter_bjorck,
        )
        self.kernel.assign(wbar)
        self.u.assign(u)
        self.sig.assign(sigma)

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        layer = Conv2DTranspose(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            activation=self.activation,
            use_bias=self.use_bias,
            **self._kwargs,
        )
        layer.build(self.input.shape)
        layer.kernel.assign(self.wbar)
        if layer.use_bias:
            layer.bias.assign(self.bias)
        return layer


@register_keras_serializable("deel-lip", "FrobeniusConv2D")
class FrobeniusConv2D(Conv2D, LipschitzLayer, Condensable):
    """
    Same as SpectralConv2D but in the case of a single output.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer=SpectralInitializer(),
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        k_coef_lip=1.0,
        **kwargs,
    ):
        if strides not in ((1, 1), [1, 1], 1):
            raise RuntimeError("FrobeniusConv2D does not support strides")
        if dilation_rate not in ((1, 1), [1, 1], 1):
            raise RuntimeError("FrobeniusConv2D does not support dilation rate")
        if padding != "same":
            raise RuntimeError("FrobeniusConv2D only supports padding='same'")
        if groups != 1:
            raise ValueError("SpectralConv2D does not support groups != 1")
        if not (
            (kernel_constraint is None)
            or isinstance(kernel_constraint, SpectralConstraint)
        ):
            raise RuntimeError(
                "only deel-lip constraints are allowed as other constraints could break"
                " 1-Lipschitz condition"
            )
        super(FrobeniusConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.set_klip_factor(k_coef_lip)
        self.wbar = None
        self._kwargs = kwargs

    def build(self, input_shape):
        super(FrobeniusConv2D, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.wbar = keras.Variable(self.kernel.value, trainable=False, name="wbar")
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        return _compute_conv_lip_factor(
            self.kernel_size, self.strides, input_shape, self.data_format
        )

    def call(self, x, training=True):
        if training:
            wbar = (
                self.kernel / K.norm(K.reshape(self.kernel, (-1,))) * self._get_coef()
            )
            self.wbar.assign(wbar)
        else:
            wbar = self.wbar

        # Compute Conv2D operation (copied from keras.layers.Conv2D)
        outputs = K.conv(
            x,
            wbar,
            strides=list(self.strides),
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

        if self.use_bias:
            if self.data_format == "channels_last":
                bias_shape = (1,) * (self.rank + 1) + (self.filters,)
            else:
                bias_shape = (1, self.filters) + (1,) * self.rank
            bias = K.reshape(self.bias, bias_shape)
            outputs += bias

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
        }
        base_config = super(FrobeniusConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def condense(self):
        wbar = self.kernel / K.norm(K.reshape(self.kernel, (-1,))) * self._get_coef()
        self.kernel.assign(wbar)

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        # call the condense function from SpectralDense as if it was from this class
        return SpectralConv2D.vanilla_export(self)
