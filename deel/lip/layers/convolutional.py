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
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D
from tensorflow.keras.utils import register_keras_serializable

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
    spectral_normalization_dw_conv,
)
from .base_layer import Condensable, LipschitzLayer
from ..utils import _padding_circular

try:
    from keras.utils import conv_utils  # in Keras for TF >= 2.6
except ModuleNotFoundError:
    from tensorflow.python.keras.utils import conv_utils  # in TF.python for TF <= 2.5


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
        raise RuntimeError("data_format not understood: " % data_format)

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
        **kwargs
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
        if not (
            (dilation_rate == (1, 1))
            or (dilation_rate == [1, 1])
            or (dilation_rate == 1)
        ):
            raise RuntimeError("SpectralConv2D does not support dilation rate")
        if padding != "same":
            raise RuntimeError("SpectralConv2D only supports padding='same'")
        super(SpectralConv2D, self).__init__(
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
            shape=tuple([1, self.kernel.shape.as_list()[-1]]),
            initializer=RandomNormal(0, 1),
            name="sn",
            trainable=False,
            dtype=self.dtype,
        )

        self.sig = self.add_weight(
            shape=tuple([1, 1]),  # maximum spectral  value
            name="sigma",
            trainable=False,
            dtype=self.dtype,
        )
        self.sig.assign([[1.0]])
        self.wbar = tf.Variable(self.kernel.read_value(), trainable=False)
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
        outputs = K.conv2d(
            x,
            wbar,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)
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
            **self._kwargs
        )
        layer.build(self.input_shape)
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
        output_padding=None,
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
        **kwargs
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
        `tf.keras.layers.Conv2DTranspose` doc.

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
            output_padding,
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
            **kwargs
        )

        if self.dilation_rate != (1, 1):
            raise ValueError("SpectralConv2DTranspose does not support dilation rate")
        if self.padding != "same":
            raise ValueError("SpectralConv2DTranspose only supports padding='same'")
        if self.output_padding is not None:
            raise ValueError(
                "SpectralConv2DTranspose only supports output_padding=None"
            )
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
            shape=tuple([1, self.kernel.shape.as_list()[2]]),
            initializer=RandomNormal(0, 1),
            name="sn",
            trainable=False,
            dtype=self.dtype,
        )

        self.sig = self.add_weight(
            shape=tuple([1, 1]),  # maximum spectral  value
            name="sigma",
            trainable=False,
            dtype=self.dtype,
        )
        self.sig.assign([[1.0]])
        self.wbar = tf.Variable(self.kernel.read_value(), trainable=False)

    def _compute_lip_coef(self, input_shape=None):
        return _compute_conv_lip_factor(
            self.kernel_size, self.strides, input_shape, self.data_format
        )

    def call(self, inputs, training=True):
        if training:
            kernel_reshaped = tf.transpose(self.kernel, [0, 1, 3, 2])
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
            wbar = tf.transpose(wbar, [0, 1, 3, 2])
            self.wbar.assign(wbar)
            self.u.assign(u)
            self.sig.assign(sigma)
        else:
            wbar = self.wbar

        # Apply conv2D_transpose operation on constrained weights
        # (code from TF/Keras 2.9.1)
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == "channels_first":
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height, width = None, None
        if inputs.shape.rank is not None:
            dims = inputs.shape.as_list()
            height = dims[h_axis]
            width = dims[w_axis]
        height = height if height is not None else inputs_shape[h_axis]
        width = width if width is not None else inputs_shape[w_axis]

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_output_length(
            height,
            kernel_h,
            padding=self.padding,
            output_padding=out_pad_h,
            stride=stride_h,
            dilation=self.dilation_rate[0],
        )
        out_width = conv_utils.deconv_output_length(
            width,
            kernel_w,
            padding=self.padding,
            output_padding=out_pad_w,
            stride=stride_w,
            dilation=self.dilation_rate[1],
        )
        if self.data_format == "channels_first":
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)
        output_shape_tensor = tf.stack(output_shape)

        outputs = K.conv2d_transpose(
            inputs,
            wbar,
            output_shape_tensor,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        if not tf.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        if self.use_bias:
            outputs = tf.nn.bias_add(
                outputs,
                self.bias,
                data_format=conv_utils.convert_data_format(self.data_format, ndim=4),
            )

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
            **self._kwargs
        )
        layer.build(self.input_shape)
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
        **kwargs
    ):
        if not ((strides == (1, 1)) or (strides == [1, 1]) or (strides == 1)):
            raise RuntimeError("FrobeniusConv2D does not support strides")
        if not (
            (dilation_rate == (1, 1))
            or (dilation_rate == [1, 1])
            or (dilation_rate == 1)
        ):
            raise RuntimeError("FrobeniusConv2D does not support dilation rate")
        if padding != "same":
            raise RuntimeError("FrobeniusConv2D only supports padding='same'")
        if not (
            (kernel_constraint is None)
            or isinstance(kernel_constraint, SpectralConstraint)
        ):
            raise RuntimeError(
                "only deellip constraints are allowed as other constraints could break"
                " 1 lipschitz condition"
            )
        super(FrobeniusConv2D, self).__init__(
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
        self.set_klip_factor(k_coef_lip)
        self.wbar = None
        self._kwargs = kwargs

    def build(self, input_shape):
        super(FrobeniusConv2D, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.wbar = tf.Variable(self.kernel.read_value(), trainable=False)
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        return _compute_conv_lip_factor(
            self.kernel_size, self.strides, input_shape, self.data_format
        )

    def call(self, x, training=True):
        if training:
            wbar = (self.kernel / tf.norm(self.kernel)) * self._get_coef()
            self.wbar.assign(wbar)
        else:
            wbar = self.wbar
        outputs = K.conv2d(
            x,
            wbar,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)
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
        wbar = self.kernel / tf.norm(self.kernel) * self._get_coef()
        self.kernel.assign(wbar)

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        # call the condense function from SpectralDense as if it was from this class
        return SpectralConv2D.vanilla_export(self)


@register_keras_serializable("deel-lip", "SpectralDepthwiseConv2D")
class SpectralDepthwiseConv2D(DepthwiseConv2D, LipschitzLayer, Condensable):
    def __init__(
        self,
        kernel_size,
        strides=(1, 1),
        padding="same",
        depth_multiplier=1,
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        k_coef_lip=1.0,
        filterwise_normalisation=True,
        orthogonal=False,
        ortho_niter=-1,
        eps_spectral=DEFAULT_EPS_SPECTRAL,
        maxiter_spectral=DEFAULT_MAXITER_SPECTRAL,
        **kwargs
    ):
        """
        This class is a DepthwiseConv2D Layer constrained to be 1-lipschitz. This is
        done by applying the power iteration algorithm for convolution.


        Args:
            kernel_size: An integer or tuple/list of 2 integers, specifying the height
                and width of the 2D convolution window. Can be a single integer to
                specify the same value for all spatial dimensions.
            strides: An integer or tuple/list of 2 integers, specifying the strides of
                the convolution along the height and width. Can be a single integer to
                specify the same value for all spatial dimensions. Current
                implementation only supports equal length strides in row and
                column dimensions. Specifying any stride value != 1 is incompatible
                with specifying any `dilation_rate` value !=1.
            padding: one of `'valid'` or `'same'` (case-insensitive). `"valid"` means
                no padding. `"same"` results in padding with zeros evenly to the
                left/right or up/down of the input such that output has the same
                height/width dimension as the input.
            depth_multiplier: The number of depthwise convolution output channels for
                each input channel. The total number of depthwise convolution output
                channels will be equal to `filters_in * depth_multiplier`.
            data_format: A string, one of `channels_last` (default) or
                `channels_first`. The ordering of the dimensions in the inputs.
                `channels_last` corresponds to inputs with shape `(batch_size, height,
                width, channels)` while `channels_first` corresponds to inputs with
                shape `(batch_size, channels, height, width)`. It defaults to the
                `image_data_format` value found in your Keras config file at
                `~/.keras/keras.json`. If you never set it, then it will be
                'channels_last'.
            dilation_rate: An integer or tuple/list of 2 integers, specifying the
                dilation rate to use for dilated convolution. Currently, specifying any
                `dilation_rate` value != 1 is incompatible with specifying any `strides`
                value != 1.
            activation: Activation function to use. If you don't specify anything, no
                activation is applied (see `keras.activations`).
            use_bias: Boolean, whether the layer uses a bias vector.
            depthwise_initializer: Initializer for the depthwise kernel matrix (see
                `keras.initializers`). If None, the default initializer
                ('glorot_uniform') will be used.
            bias_initializer: Initializer for the bias vector (see
                `keras.initializers`). If None, the default initializer ('zeros') will
                be used.
            depthwise_regularizer: Regularizer function applied to the depthwise
                kernel matrix (see `keras.regularizers`).
            bias_regularizer: Regularizer function applied to the bias vector (see
                `keras.regularizers`).
            activity_regularizer: Regularizer function applied to the output of the
                layer (its 'activation') (see `keras.regularizers`).
            depthwise_constraint: Constraint function applied to the depthwise kernel
                matrix (see `keras.constraints`).
            bias_constraint: Constraint function applied to the bias vector (see
                `keras.constraints`).
            k_coef_lip: Lipschitz constant to ensure.
            filterwise_normalisation: when True normalize each filter independently,
                else normalise the filters globally. Both approaches yield k_coef_lip
                convolutions.

        Input shape:
            4D tensor with shape: `[batch_size, channels, rows, cols]` if
                data_format='channels_first'
            or 4D tensor with shape: `[batch_size, rows, cols, channels]` if
                data_format='channels_last'.

        Output shape:
            4D tensor with shape: `[batch_size, channels * depth_multiplier, new_rows,
                new_cols]` if `data_format='channels_first'`
                or 4D tensor with shape: `[batch_size,
                new_rows, new_cols, channels * depth_multiplier]` if
                `data_format='channels_last'`. `rows` and `cols` values might have
                changed due to padding.

        Returns:
            A tensor of rank 4 representing
            `activation(depthwiseconv2d(inputs, kernel) + bias)`.

        Raises:
            ValueError: if `padding` is "causal".
            ValueError: when both `strides` > 1 and `dilation_rate` > 1.

        """
        self.ortho_niter = ortho_niter
        self.orthogonal = orthogonal
        if orthogonal and not filterwise_normalisation:
            raise ValueError(
                "filterwise_normalisation must be True when "
                "orthogonalization is enabled"
            )
        if eps_spectral <= 0:
            raise ValueError("eps_spectral has to be > 0")
        if maxiter_spectral <= 0:
            raise ValueError("maxiter_spectral has to be > 0")
        self.eps_spectral = eps_spectral
        self.maxiter_spectral = maxiter_spectral
        if data_format == "channels_first":
            raise ValueError(
                "channels first data format is not supported for depthwise conv"
            )
        self.filterwise_normalisation = filterwise_normalisation
        if not (
            (dilation_rate == (1, 1))
            or (dilation_rate == [1, 1])
            or (dilation_rate == 1)
        ):
            raise ValueError("SpectralDepthwiseConv2D does not support dilation rate")
        if depth_multiplier != 1:
            raise ValueError(
                "SpectralDepthwiseConv2D does not support depth multiplier"
            )
        self.pad = lambda x: x
        self.old_padding = padding
        if padding.lower() in ["same", "valid"]:
            self.pad = lambda x: x
        elif padding.upper() in ["CONSTANT", "REFLECT", "SYMMETRIC"]:
            p_vert, p_hor = kernel_size[0] // 2, kernel_size[1] // 2
            paddings = [[0, 0], [p_vert, p_vert], [p_hor, p_hor], [0, 0]]
            mode = padding.upper()
            self.pad = lambda t: tf.pad(t, paddings, mode)
            padding = "valid"
        elif padding.lower() in ["circular"]:
            padding = "valid"
            p_vert, p_hor = kernel_size[0] // 2, kernel_size[1] // 2
            self.pad = lambda t: _padding_circular(t, (p_vert, p_hor))
        else:
            raise RuntimeError(
                "padding must be in same, valid, CONSTANT, REFLECT, "
                "SYMMETRIC, circular"
            )
        super(SpectralDepthwiseConv2D, self).__init__(
            kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self._kwargs = kwargs
        self.set_klip_factor(k_coef_lip)

    def build(self, input_shape):
        super(SpectralDepthwiseConv2D, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self._input_shape = input_shape
        if self.data_format == "channels_first":
            c, w, h = input_shape[-3], input_shape[-2], input_shape[-1]
            u_shape = (
                1,
                c,
                min(w, 3 * self.kernel_size[0] + 1),
                min(h, 3 * self.kernel_size[1] + 1),
            )
        else:
            w, h, c = input_shape[-3], input_shape[-2], input_shape[-1]
            u_shape = (
                1,
                min(w, 3 * self.kernel_size[0] + 1),
                min(h, 3 * self.kernel_size[1] + 1),
                c,
            )
        self.u = self.add_weight(
            shape=u_shape,
            initializer=RandomNormal(0, 1),
            name="u",
            trainable=False,
            dtype=self.dtype,
        )
        sig_shape = (1, 1, c, 1) if self.filterwise_normalisation else (1, 1, 1, 1)
        self.sig = self.add_weight(
            shape=sig_shape,  # maximum spectral  value
            name="sigma",
            trainable=False,
            dtype=self.dtype,
            initializer="ones",
        )
        self.wbar = tf.Variable(
            tf.cast(self.depthwise_kernel.read_value(), self.dtype),
            dtype=self.dtype,
            name="wbar",
            trainable=False,
        )
        wbar, u, sig = spectral_normalization_dw_conv(
            self.depthwise_kernel,
            self.u,
            self.strides,
            None if self.old_padding == "same" else self.pad,
            self.filterwise_normalisation,
            self.eps_spectral,
            self.maxiter_spectral,
        )
        self.wbar.assign(wbar)
        self.u.assign(u)
        self.sig.assign(sig)
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        return 1.0

    @tf.function
    def call(self, x, training=True):
        if training:
            if self.orthogonal:
                dw_kernel = 0.5 * (
                    self.depthwise_kernel - self.depthwise_kernel[::-1, ::-1, :]
                )
            else:
                dw_kernel = self.depthwise_kernel
            wbar, u, sig = spectral_normalization_dw_conv(
                dw_kernel,
                self.u,
                self.strides,
                None if self.old_padding == "same" else self.pad,
                self.filterwise_normalisation,
                self.eps_spectral,
                self.maxiter_spectral,
            )
            self.wbar.assign(wbar)
            self.u.assign(u)
            self.sig.assign(sig)
        else:
            wbar = self.wbar

        if self.orthogonal:
            outputs = x
            for i in range(1, self.ortho_niter):
                x = self.pad(x)
                x = K.depthwise_conv2d(
                    x,
                    wbar * self._get_coef(),
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate,
                )
                x = x / i
                outputs += x
        else:
            x = self.pad(x)
            outputs = K.depthwise_conv2d(
                x,
                wbar * self._get_coef(),
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
            )
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "filterwise_normalisation": self.filterwise_normalisation,
            "eps_spectral": self.eps_spectral,
            "maxiter_spectral": self.maxiter_spectral,
        }
        base_config = super(SpectralDepthwiseConv2D, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        config["padding"] = self.old_padding
        return config

    def condense(self):
        wbar, u, sigma = spectral_normalization_dw_conv(
            self.depthwise_kernel,
            self.u,
            self.strides[-1],
            None if self.old_padding == "same" else self.pad,
            self.eps_spectral,
            self.maxiter_spectral,
        )
        self.kernel.assign(wbar)
        self.u.assign(u)
        self.sig.assign(sigma)

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        layer = DepthwiseConv2D(
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            depth_multiplier=self.depth_multiplier,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            use_bias=self.use_bias,
            depthwise_initializer=self.depthwise_initializer,
            bias_initializer=self.bias_initializer,
            depthwise_regularizer=self.depthwise_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            depthwise_constraint=self.depthwise_constraint,
            bias_constraint=self.bias_constraint,
            **self._kwargs
        )
        layer.build(self.input_shape)
        layer.depthwise_kernel.assign(self.wbar * self._get_coef())
        if self.use_bias:
            layer.bias.assign(self.bias)
        return layer
