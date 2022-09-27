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

import abc

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
import tensorflow.keras.layers as keraslayers

from .constraints import SpectralConstraint
from .initializers import SpectralInitializer
from .normalizers import (
    DEFAULT_EPS_BJORCK,
    DEFAULT_EPS_SPECTRAL,
    reshaped_kernel_orthogonalization,
    DEFAULT_BETA_BJORCK,
)
from tensorflow.keras.utils import register_keras_serializable

try:
    from keras.utils import conv_utils  # in Keras for TF >= 2.6
except ModuleNotFoundError:
    from tensorflow.python.keras.utils import conv_utils  # in TF.python for TF <= 2.5


class LipschitzLayer(abc.ABC):
    """
    This class allow to set lipschitz factor of a layer. Lipschitz layer must inherit
    this class to allow user to set the lipschitz factor.

    Warning:
         This class only regroup useful functions when developing new Lipschitz layers.
         But it does not ensure any property about the layer. This means that
         inheriting from this class won't ensure anything about the lipschitz constant.
    """

    k_coef_lip = 1.0
    """variable used to store the lipschitz factor"""
    coef_lip = None
    """
    define correction coefficient (ie. Lipschitz bound ) of the layer
    ( multiply the output of the layer by this constant )
    """

    def set_klip_factor(self, klip_factor):
        """
        Allow to set the lipschitz factor of a layer.

        Args:
            klip_factor: the Lipschitz factor the user want to ensure.

        Returns:
            None

        """
        self.k_coef_lip = klip_factor

    @abc.abstractmethod
    def _compute_lip_coef(self, input_shape=None):
        """
        Some layers (like convolution) cannot ensure a strict lipschitz constant (as
        the Lipschitz factor depends on the input data). Those layers then rely on the
        computation of a bounding factor. This function allow to compute this factor.

        Args:
            input_shape: the shape of the input of the layer.

        Returns:
            the bounding factor.

        """
        pass

    def _init_lip_coef(self, input_shape):
        """
        Initialize the lipschitz coefficient of a layer.

        Args:
            input_shape: the layers input shape

        Returns:
            None

        """
        self.coef_lip = self._compute_lip_coef(input_shape)

    def _get_coef(self):
        """
        Returns:
            the multiplicative coefficient to be used on the result in order to ensure
            k-Lipschitzity.
        """
        if self.coef_lip is None:
            raise RuntimeError("compute_coef must be called before calling get_coef")
        return self.coef_lip * self.k_coef_lip


class Condensable(abc.ABC):
    """
    Some Layers don't optimize directly the kernel, this means that the kernel stored
    in the layer is not the kernel used to make predictions (called W_bar), to address
    this, these layers can implement the condense() function that make self.kernel equal
    to W_bar.

    This operation also allow the turn the lipschitz layer to it keras equivalent ie.
    The Dense layer that have the same predictions as the trained SpectralDense.
    """

    @abc.abstractmethod
    def condense(self):
        """
        The condense operation allow to overwrite the kernel and ensure that other
        variables are still consistent.

        Returns:
            None

        """
        pass

    @abc.abstractmethod
    def vanilla_export(self):
        """
        This operation allow to turn this Layer to it's super type, easing storage and
        serving.

        Returns:
             self as super type

        """
        pass


def _check_RKO_params(eps_spectral, eps_bjorck, beta_bjorck):
    """Assert that the RKO hyper-parameters are supported values."""
    if eps_spectral <= 0:
        raise ValueError("eps_spectral has to be > 0")
    if (eps_bjorck is not None) and (eps_bjorck <= 0.0):
        raise ValueError("eps_bjorck must be > 0")
    if (beta_bjorck is not None) and not (0.0 < beta_bjorck <= 0.5):
        raise ValueError("beta_bjorck must be in ]0, 0.5]")


@register_keras_serializable("deel-lip", "SpectralDense")
class SpectralDense(keraslayers.Dense, LipschitzLayer, Condensable):
    def __init__(
        self,
        units,
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
        **kwargs
    ):
        """
        This class is a Dense Layer constrained such that all singular of it's kernel
        are 1. The computation based on Bjorck algorithm.
        The computation is done in two steps:

        1. reduce the larget singular value to 1, using iterated power method.
        2. increase other singular values to 1, using Bjorck algorithm.

        Args:
            units: Positive integer, dimensionality of the output space.
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
            kernel_constraint: Constraint function applied to
                the `kernel` weights matrix.
            bias_constraint: Constraint function applied to the bias vector.
            k_coef_lip: lipschitz constant to ensure
            eps_spectral: stopping criterion for the iterative power algorithm.
            eps_bjorck: stopping criterion Bjorck algorithm.
            beta_bjorck: beta parameter in bjorck algorithm.

        Input shape:
            N-D tensor with shape: `(batch_size, ..., input_dim)`.
            The most common situation would be
            a 2D input with shape `(batch_size, input_dim)`.

        Output shape:
            N-D tensor with shape: `(batch_size, ..., units)`.
            For instance, for a 2D input with shape `(batch_size, input_dim)`,
            the output would have shape `(batch_size, units)`.

        This documentation reuse the body of the original keras.layers.Dense doc.
        """
        super(SpectralDense, self).__init__(
            units=units,
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
        _check_RKO_params(eps_spectral, eps_bjorck, beta_bjorck)
        self.eps_spectral = eps_spectral
        self.beta_bjorck = beta_bjorck
        self.eps_bjorck = eps_bjorck
        self.u = None
        self.sig = None
        self.wbar = None
        self.built = False

    def build(self, input_shape):
        super(SpectralDense, self).build(input_shape)
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
            initializer=tf.keras.initializers.ones,
            name="sigma",
            trainable=False,
            dtype=self.dtype,
        )
        self.sig.assign([[1.0]])
        self.wbar = tf.Variable(self.kernel.read_value(), trainable=False)
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        return 1.0  # this layer don't require a corrective factor

    @tf.function
    def call(self, x, training=True):
        if training:
            wbar, u, sigma = reshaped_kernel_orthogonalization(
                self.kernel,
                self.u,
                self._get_coef(),
                self.eps_spectral,
                self.eps_bjorck,
                self.beta_bjorck,
            )
            self.wbar.assign(wbar)
            self.u.assign(u)
            self.sig.assign(sigma)
        else:
            wbar = self.wbar
        outputs = tf.matmul(x, wbar)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "eps_spectral": self.eps_spectral,
            "eps_bjorck": self.eps_bjorck,
            "beta_bjorck": self.beta_bjorck,
        }
        base_config = super(SpectralDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def condense(self):
        wbar, u, sigma = reshaped_kernel_orthogonalization(
            self.kernel,
            self.u,
            self._get_coef(),
            self.eps_spectral,
            self.eps_bjorck,
            self.beta_bjorck,
        )
        self.kernel.assign(wbar)
        self.u.assign(u)
        self.sig.assign(sigma)

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        layer = keraslayers.Dense(
            units=self.units,
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
class SpectralConv2D(keraslayers.Conv2D, LipschitzLayer, Condensable):
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
        )
        self.kernel.assign(wbar)
        self.u.assign(u)
        self.sig.assign(sigma)

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        layer = keraslayers.Conv2D(
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
class SpectralConv2DTranspose(keraslayers.Conv2DTranspose, LipschitzLayer, Condensable):
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
        )
        self.kernel.assign(wbar)
        self.u.assign(u)
        self.sig.assign(sigma)

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        layer = keraslayers.Conv2DTranspose(
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


@register_keras_serializable("deel-lip", "FrobeniusDense")
class FrobeniusDense(keraslayers.Dense, LipschitzLayer, Condensable):
    """
    Identical and faster than a SpectralDense in the case of a single output. In the
    multi-neurons setting, this layer can be used:
    - as a classical Frobenius Dense normalization (disjoint_neurons=False)
    - as a stacking of 1 lipschitz independent neurons (each output is 1-lipschitz,
    but the no orthogonality is enforced between outputs )  (disjoint_neurons=True).

    Warning :
        default is disjoint_neurons = True
    """

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer=SpectralInitializer(),
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        disjoint_neurons=True,
        k_coef_lip=1.0,
        **kwargs
    ):
        super().__init__(
            units=units,
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
        self.disjoint_neurons = disjoint_neurons
        self.axis_norm = None
        self.wbar = None
        if self.disjoint_neurons:
            self.axis_norm = 0
        self._kwargs = kwargs

    def build(self, input_shape):
        super(FrobeniusDense, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.wbar = tf.Variable(self.kernel.read_value(), trainable=False)
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        return 1.0

    def call(self, x, training=True):
        if training:
            wbar = (
                self.kernel
                / tf.norm(self.kernel, axis=self.axis_norm)
                * self._get_coef()
            )
            self.wbar.assign(wbar)
        else:
            wbar = self.wbar
        outputs = tf.matmul(x, wbar)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "disjoint_neurons": self.disjoint_neurons,
        }
        base_config = super(FrobeniusDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def condense(self):
        wbar = (
            self.kernel / tf.norm(self.kernel, axis=self.axis_norm) * self._get_coef()
        )
        self.kernel.assign(wbar)

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        layer = keraslayers.Dense(
            units=self.units,
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


@register_keras_serializable("deel-lip", "FrobeniusConv2D")
class FrobeniusConv2D(keraslayers.Conv2D, LipschitzLayer, Condensable):
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


@register_keras_serializable("deel-lip", "ScaledAveragePooling2D")
class ScaledAveragePooling2D(keraslayers.AveragePooling2D, LipschitzLayer):
    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        k_coef_lip=1.0,
        **kwargs
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
            **kwargs
        )
        self.set_klip_factor(k_coef_lip)
        self._kwargs = kwargs

    def build(self, input_shape):
        super(ScaledAveragePooling2D, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        return np.sqrt(np.prod(np.asarray(self.pool_size)))

    def call(self, x, training=True):
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
        **kwargs
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
            **kwargs
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

    def call(self, x, training=True):
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

    def call(self, x, training=True):
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
            raise RuntimeError("data format not understood: %s" % self.data_format)
        return lip_coef

    def call(self, x, training=True):
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
            Anil et al. https://arxiv.org/abs/1911.00937

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

    def build(self, input_shape):
        return super(InvertibleDownSampling, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs = super(InvertibleDownSampling, self).call(inputs, **kwargs)
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
            Anil et al. https://arxiv.org/abs/1911.00937

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

    def build(self, input_shape):
        return super(InvertibleUpSampling, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.data_format == "channels_first":
            # convert to channels_first
            inputs = tf.transpose(inputs, [0, 2, 3, 1])
        # from shape (bs, w, h, c*pw*ph) to (bs, w, h, pw, ph, c)
        bs, w, h = inputs.shape[:-1]
        (
            pw,
            ph,
        ) = self.pool_size
        c = inputs.shape[-1] // (pw * ph)
        print(c)
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
