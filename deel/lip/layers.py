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
import warnings

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
from .normalizers import (
    spectral_normalization_conv,
)
from .regularizers import LorthRegularizer
from .utils import _padding_circular


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
        self.eps_spectral = eps_spectral
        self.beta_bjorck = beta_bjorck
        if (self.beta_bjorck is not None) and (
            not ((self.beta_bjorck <= 0.5) and (self.beta_bjorck > 0.0))
        ):
            raise RuntimeError("beta_bjorck must be in ]0, 0.5]")
        self.eps_bjorck = eps_bjorck
        if (self.eps_bjorck is not None) and (not self.eps_bjorck > 0.0):
            raise RuntimeError("eps_bjorck must be in > 0")
        self.u = None
        self.sig = None
        self.wbar = None
        self.built = False
        if self.eps_spectral < 0:
            raise RuntimeError("eps_spectral has to be > 0")

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


@register_keras_serializable("deel-lip", "PadConv2D")
class PadConv2D(keraslayers.Conv2D, Condensable):
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
        This class is a Conv2D Layer with paramtrized padding
        Since Conv2D layer ony support `"same"` and `"valid"` padding,
        this layer will enable other type of padding, such as
        `"constant"`, `"symmetric"`, `"reflect"` or `"circular"`

        Args:
            Same args as the body of the original keras.layers.Conv2D, except for
            paddiing accepts
            padding: one of `"same"`, `"valid"` `"constant"`, `"symmetric"`,
            `"reflect"` or `"circular"` (case-insensitive).

        """
        self.pad = lambda x: x
        self.old_padding = padding
        self.internal_input_shape = None
        if not padding.lower() in ["same"]:  # same is directly processed in Conv2D
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
        if self.old_padding.upper() in ["CONSTANT", "REFLECT", "SYMMETRIC"]:
            self.padding_size = [
                self.kernel_size[0] // 2,
                self.kernel_size[1] // 2,
            ]  # require kernel_size as a list -> done in Conv2D::_init_
            paddings = [
                [0, 0],
                [self.padding_size[0], self.padding_size[0]],
                [self.padding_size[1], self.padding_size[1]],
                [0, 0],
            ]
            self.pad = lambda t: tf.pad(t, paddings, self.old_padding)
        if self.old_padding.lower() in ["circular"]:
            self.padding_size = [self.kernel_size[0] // 2, self.kernel_size[1] // 2]
            self.pad = lambda t: _padding_circular(t, self.padding_size)

    def compute_padded_shape(self, input_shape, padding_size):
        if isinstance(input_shape, tf.TensorShape):
            internal_input_shape = input_shape.as_list()
        else:
            internal_input_shape = list(input_shape)

        if self.data_format == "channels_last":
            first_layer = 1
        else:
            first_layer = 2
        for index, pad in enumerate(padding_size):
            internal_input_shape[index + first_layer] += 2 * pad
        internal_input_shape = tf.TensorShape(internal_input_shape)
        return internal_input_shape

    def build(self, input_shape):
        self.internal_input_shape = self.compute_padded_shape(
            input_shape, self.padding_size
        )
        print("build internal_input_shape ", self.internal_input_shape)
        super(PadConv2D, self).build(self.internal_input_shape)

    def compute_output_shape(self, input_shape):
        return super(PadConv2D, self).compute_output_shape(self.internal_input_shape)

    def call(self, x, training=True):
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
            layer_type = keraslayers.Conv2D
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
            raise RuntimeError("NormalizedConv does not support dilation rate")
        if padding != "same":
            raise RuntimeError("NormalizedConv only support padding='same'")
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
        self.eps_spectral = eps_spectral
        self.beta_bjorck = beta_bjorck
        if (self.beta_bjorck is not None) and (
            not ((self.beta_bjorck <= 0.5) and (self.beta_bjorck > 0.0))
        ):
            raise RuntimeError("beta_bjorck must be in ]0, 0.5]")
        self.eps_bjorck = eps_bjorck
        if (self.eps_bjorck is not None) and (not self.eps_bjorck > 0.0):
            raise RuntimeError("eps_bjorck must be in > 0")
        if self.eps_spectral < 0:
            raise RuntimeError("eps_spectral has to be > 0")

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
        # According to the file lipschitz_CNN.pdf
        stride = np.prod(self.strides)
        k1 = self.kernel_size[0]
        k1_div2 = (k1 - 1) / 2
        k2 = self.kernel_size[1]
        k2_div2 = (k2 - 1) / 2
        if self.data_format == "channels_last":
            h = input_shape[-3]
            w = input_shape[-2]
        elif self.data_format == "channels_first":
            h = input_shape[-2]
            w = input_shape[-1]
        else:
            raise RuntimeError("data_format not understood: " % self.data_format)
        if stride == 1:
            coefLip = np.sqrt(
                (w * h)
                / (
                    (k1 * h - k1_div2 * (k1_div2 + 1))
                    * (k2 * w - k2_div2 * (k2_div2 + 1))
                )
            )
        else:
            sn1 = self.strides[0]
            sn2 = self.strides[1]
            coefLip = np.sqrt(1.0 / (np.ceil(k1 / sn1) * np.ceil(k2 / sn2)))
        return coefLip

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
                circular_paddings=self.padding_size,
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
            raise RuntimeError("NormalizedConv does not support strides")
        if not (
            (dilation_rate == (1, 1))
            or (dilation_rate == [1, 1])
            or (dilation_rate == 1)
        ):
            raise RuntimeError("NormalizedConv does not support dilation rate")
        if padding != "same":
            raise RuntimeError("NormalizedConv only support padding='same'")
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
        return SpectralConv2D._compute_lip_coef(self, input_shape)

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
        wbar = (self.kernel / tf.norm(self.kernel)) * self._get_coef()
        self.wbar.assign(wbar)
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
            raise RuntimeError("ScaledAveragePooling2D only support padding='valid'")
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
            raise RuntimeError("NormalizedConv only support padding='valid'")
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
