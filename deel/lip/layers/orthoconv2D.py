# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module proposes the OrthoConv2D layer which is a Keras convolutional layer where
the orthogonality is handled by a Lorth regularization term.
"""

import warnings

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import register_keras_serializable
import tensorflow as tf

from ..normalizers import DEFAULT_EPS_SPECTRAL, spectral_normalization_conv
from ..regularizers import LorthRegularizer
from ..initializers import LorthInitializer
from ..constraints import LorthConstraint
from .base_layer import Condensable, LipschitzLayer
from .unconstrained import PadConv2D

from ..utils_lorth import (
    Lorth2Dgrad,
    DEFAULT_NITER_LORTHGRAD,
)


@register_keras_serializable("deel-lip", "OrthoConv2D")
class OrthoConv2D(PadConv2D, LipschitzLayer, Condensable):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="circular",
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
        k_coef_lip=1.0,
        eps_spectral=DEFAULT_EPS_SPECTRAL,
        regul_lorth=10.0,
        **kwargs
    ):
        """
        This class is a Conv2D Layer regularized such that all singular of it's kernel
        are 1. The computation is based on LorthRegularizer and requires a circular
        padding (added in the layer).
        The computation is done in three steps:
        1. apply a circular padding for ensuring 'same' configuration.
        2. apply conv2D.
        3. regularize kernel with Lorth Regul.
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
            padding: `"circular"` ONLY (case-insensitive).
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
                the `kernel` weights matrix (should be None and will be set).
            bias_regularizer: Regularizer function applied to the bias vector.
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation")..
            kernel_constraint: Constraint function applied to the kernel matrix.
            bias_constraint: Constraint function applied to the bias vector.
            k_coef_lip: lipschitz constant to ensure.
            regul_lorth: float : weight of the orthogonalization regularization.
        This documentation reuse the body of the original keras.layers.Conv2D doc.
        """
        if padding != "circular":
            raise RuntimeError(
                "OrthoConv2D only support padding='circular' implemented in the "
                "layer "
            )
        if kernel_regularizer is not None:
            raise RuntimeError(
                "OrthoConv2D define the kernel_regularizer (should be None)"
            )

        if isinstance(strides, int):
            self.strides = (strides,) * 2
        else:
            self.strides = tuple(strides)

        self.eps_spectral = eps_spectral
        regul_lip_conv = self.init_regul_lorth(regul_lorth, self.strides[0])

        super(OrthoConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=self.strides,
            padding=padding,  # padding taken into account in PadConv2D
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regul_lip_conv,  # internal value
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self._kwargs = kwargs
        self.set_klip_factor(k_coef_lip)
        self.sig = None
        self.u = None
        self.spectral_input_shape = None
        self.ro_case = True
        self.built = False

    def init_regul_lorth(self, regul_lorth, stride):
        self.regul_lorth = regul_lorth
        if regul_lorth < 0:
            raise RuntimeError(
                "OrthoConv2D requires a  positive regularization factor " "regul_lorth"
            )
        if regul_lorth == 0:
            if self.eps_spectral > 0:
                warnings.warn("No Lorth Regularization, spectral normalization only")
            else:
                warnings.warn("No constraint regul_lorth==0 and eps_spectral==0")

        regul_lip_conv = None
        if regul_lorth > 0:
            regul_lip_conv = LorthRegularizer(
                kernel_shape=None,
                stride=stride,
                lambda_lorth=regul_lorth,
                flag_deconv=False,
            )
        return regul_lip_conv

    def init_spectral_norm(self):
        if self.eps_spectral <= 0:
            return
        (R0, R, C, M) = self.kernel.shape
        stride = self.strides[0]
        # Compute minimal N
        r = R // 2
        if r < 1:
            N = 5
        else:
            N = 4 * r + 1
            if stride > 1:
                N = int(0.5 + N / stride)

        if C * stride**2 > M:
            self.spectral_input_shape = (N, N, M)
            self.ro_case = True
        else:
            self.spectral_input_shape = (stride * N, stride * N, C)
            self.ro_case = False
        self.u = self.add_weight(
            shape=(1,) + self.spectral_input_shape,
            initializer=RandomNormal(-1, 1),
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

    def build(self, input_shape):
        super(OrthoConv2D, self).build(input_shape)
        self._init_lip_coef(input_shape)
        if self.kernel_regularizer is not None:
            self.kernel_regularizer._set_kernel_shape(self.kernel.shape)
        self.init_spectral_norm()
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        return 1.0  # this layer don't require a corrective factor

    def call(self, x, training=None):
        if training and self.eps_spectral > 0:
            W_bar, _u, sigma = spectral_normalization_conv(
                self.kernel,
                self.u,
                stride=self.strides[0],
                conv_first=not self.ro_case,
                pad_func=self.pad,
                eps=self.eps_spectral,
            )
            self.sig.assign([[sigma]])
            self.u.assign(_u)
        else:
            if self.sig is not None:
                W_bar = self.kernel / self.sig
            else:
                W_bar = self.kernel
        kernel = self.kernel
        self.kernel = W_bar
        outputs = super(OrthoConv2D, self).call(x)
        self.kernel = kernel
        return outputs

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "eps_spectral": self.eps_spectral,
            "regul_lorth": self.regul_lorth,
            "kernel_regularizer": None,  # overwrite the kernel regul to None
        }
        base_config = super(OrthoConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def condense(self):
        if self.sig is not None:
            new_w = self.kernel / self.sig.numpy()
            self.kernel.assign(new_w)
            self.sig.assign([[1.0]])
        return

    def vanilla_export(self):
        layer = super(OrthoConv2D, self).vanilla_export()
        kernel = self.kernel
        if self.sig is not None:
            kernel = kernel / self.sig.numpy()

        layer.kernel.assign(kernel.numpy() * self._get_coef())
        if self.use_bias:
            layer.bias.assign(self.bias.numpy())
        return layer


@register_keras_serializable("deel-lip", "OrthoConv2DProj")
class OrthoConv2DProj(PadConv2D, LipschitzLayer, Condensable):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="circular",
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=LorthInitializer,
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        k_coef_lip=1.0,
        eps_spectral=DEFAULT_EPS_SPECTRAL,
        niter_newton=DEFAULT_NITER_LORTHGRAD,
        regul_lorth=0.0,
        **kwargs
    ):
        """
        This class is a Conv2D Layer regularized such that all singular of it's kernel
        are 1. The computation is based on LorthRegularizer and requires a circular
        padding (added in the layer).
        The computation is done in three steps:
        1. apply a circular padding for ensuring 'same' configuration.
        2. apply conv2D.
        3. regularize kernel with Lorth Regul.
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
            padding: `"circular"` ONLY (case-insensitive).
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
                the `kernel` weights matrix (should be None and will be set).
            bias_regularizer: Regularizer function applied to the bias vector.
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation")..
            kernel_constraint: Constraint function applied to the kernel matrix.
            bias_constraint: Constraint function applied to the bias vector.
            k_coef_lip: lipschitz constant to ensure.
            regul_lorth: float : weight of the orthogonalization regularization.
        This documentation reuse the body of the original keras.layers.Conv2D doc.
        """
        if padding != "circular":
            raise RuntimeError(
                "OrthoConv2D only support padding='circular' implemented in the "
                "layer "
            )
        if kernel_regularizer is not None:
            raise RuntimeError(
                "OrthoConv2DProj could not use a  kernel_regularizer (should be None)"
            )
        if kernel_constraint is not None:
            raise RuntimeError(
                "OrthoConv2DProj define the kernel_constraint (should be None)"
            )
        if isinstance(strides, int):
            self.strides = (strides,) * 2
        else:
            self.strides = tuple(strides)

        kernel_constraint = LorthConstraint(kernel_shape=None, stride=self.strides[0])

        self.eps_spectral = eps_spectral
        self.niter_newton = niter_newton
        regul_lip_conv = None  # self.init_regul_lorth(regul_lorth, self.strides[0])

        super(OrthoConv2DProj, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=self.strides,
            padding=padding,  # padding taken into account in PadConv2D
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regul_lip_conv,  # internal value
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self._kwargs = kwargs
        self.set_klip_factor(k_coef_lip)
        self.sig = None
        self.u = None
        self.spectral_input_shape = None
        self.ro_case = True
        self.lorthg = None
        self.built = False

    def init_regul_lorth(self, regul_lorth, stride):
        self.regul_lorth = regul_lorth
        if regul_lorth < 0:
            raise RuntimeError(
                "OrthoConv2D requires a  positive regularization factor " "regul_lorth"
            )
        if regul_lorth == 0:
            if self.eps_spectral > 0:
                warnings.warn("No Lorth Regularization, spectral normalization only")
            else:
                warnings.warn("No constraint regul_lorth==0 and eps_spectral==0")

        regul_lip_conv = None
        assert regul_lorth == 0.0
        if regul_lorth > 0:
            regul_lip_conv = LorthRegularizer(
                kernel_shape=None,
                stride=stride,
                lambda_lorth=regul_lorth,
                flag_deconv=False,
            )
        return regul_lip_conv

    def init_spectral_norm(self):
        if self.eps_spectral <= 0:
            return
        (R0, R, C, M) = self.kernel.shape
        stride = self.strides[0]
        # Compute minimal N
        r = R // 2
        if r < 1:
            N = 5
        else:
            N = 4 * r + 1
            if stride > 1:
                N = int(0.5 + N / stride)

        if C * stride**2 > M:
            self.spectral_input_shape = (N, N, M)
            self.ro_case = True
        else:
            self.spectral_input_shape = (stride * N, stride * N, C)
            self.ro_case = False
        self.u = self.add_weight(
            shape=(1,) + self.spectral_input_shape,
            initializer=RandomNormal(-1, 1),
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

    def build(self, input_shape):
        super(OrthoConv2DProj, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.wbar = tf.Variable(self.kernel.read_value(), trainable=False)
        if self.kernel_regularizer is not None:
            self.kernel_regularizer.set_kernel_shape(self.kernel.shape)
        if self.kernel_constraint is not None:
            self.kernel_constraint.set_kernel_shape(self.kernel.shape)

        self.lorthg = Lorth2Dgrad(
            kernel_shape=self.kernel.shape,
            stride=self.strides[0],
            niter_newton=self.niter_newton,
        )

        self.init_spectral_norm()
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        return 1.0  # this layer don't require a corrective factor

    def call(self, x, training=None):
        if training and self.eps_spectral > 0:
            W_bar, _u, sigma = spectral_normalization_conv(
                self.kernel,
                self.u,
                stride=self.strides[0],
                conv_first=not self.ro_case,
                pad_func=self.pad,
                eps=self.eps_spectral,
            )
            self.sig.assign([[sigma]])
            self.u.assign(_u)

            wbar = self.lorthg.lorthGradient_orthogonalization(W_bar, verbose=False)
            self.wbar.assign(wbar)

            W_bar = wbar
        else:
            if self.wbar is not None:
                W_bar = self.wbar
            else:
                W_bar = self.kernel
        kernel = self.kernel
        self.kernel = W_bar
        outputs = super(OrthoConv2DProj, self).call(x)
        self.kernel = kernel
        return outputs

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "eps_spectral": self.eps_spectral,
            "niter_newton": self.niter_newton,
            "regul_lorth": self.regul_lorth,
            "kernel_regularizer": None,  # overwrite the kernel regul to None
            "kernel_constraint": None,  # overwrite the kernel constraint to None
        }
        base_config = super(OrthoConv2DProj, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def condense(self):
        if self.sig is not None:
            if self.wbar is not None:
                new_w = self.wbar
            else:
                new_w = self.kernel / self.sig.numpy()
            self.kernel.assign(new_w)
            self.sig.assign([[1.0]])
        return

    def vanilla_export(self):
        layer = super(OrthoConv2DProj, self).vanilla_export()
        kernel = self.kernel
        if self.wbar is not None:
            kernel = self.wbar
        else:
            kernel = kernel / self.sig.numpy()

        layer.kernel.assign(kernel.numpy() * self._get_coef())
        if self.use_bias:
            layer.bias.assign(self.bias.numpy())
        return layer
