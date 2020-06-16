"""
This module extends original keras layers, in order to add k lipschitz constraint via reparametrization.
Currently, are implemented:

* Dense layer: as SpectralDense ( and as FrobeniusDense when the layer has a single output )

* Conv2D layer: as SpectralConv2D ( and as FrobeniusConv2D when the layer has a single output )

* AveragePooling: as ScaledAveragePooling

* GlobalAveragePooling2D: as ScaledGlobalAveragePooling2D

* MaxPooling2D: as ScaledMaxPooling2D

By default the layers are 1 Lipschitz almost everywhere, which is efficient for wasserstein distance estimation. However
for other problems (such as adversarial robustness ) the user may want to use layers that are at most 1 lipschitz, this
can be done by setting the param `niter_bjorck=0`.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
    MaxPooling2D as KerasMaxPooling2D,
)
from .constraints import SpectralNormalizer, BjorckNormalizer
from .initializers import BjorckInitializer, SpectralInitializer
from .normalizers import (
    DEFAULT_NITER_BJORCK,
    DEFAULT_NITER_SPECTRAL,
    DEFAULT_NITER_SPECTRAL_INIT,
)
from .normalizers import bjorck_normalization, spectral_normalization
from .utils import deel_export


class LipschitzLayer:
    """
    This class allow to set lipschitz factor of a layer. Lipschitz layer must inherit this class to allow user to set
    the lipschitz factor.

    Warning:
         This class only regroup useful functions when developing new Lipschitz layers. But it does not ensure any
         property about the layer. This means that inheriting from this class won't ensure anything about the lipschitz
         constant.
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

        Returns: None

        """
        self.k_coef_lip = klip_factor

    def _compute_lip_coef(self, input_shape=None):
        """
        Some layers ( like convolution ) cannot ensure a strict lipschitz constant ( as the Lipschitz factor depends on
        the input data ). Those layers then rely on the computation of a bounding factor. This function allow to
        compute this factor.

        Args:
            input_shape: the shape of the input of the layer.

        Returns: the bounding factor.

        """
        raise NotImplementedError(
            "classes that inherits from LipschitzLayer must implement the compute_coef function"
        )

    def _init_lip_coef(self, input_shape):
        """
        Initialize the lipschitz coefficient of a layer.

        Args:
            input_shape: the layers input shape

        Returns: None

        """
        self.coef_lip = self._compute_lip_coef(input_shape)

    def _get_coef(self):
        """
        Returns: the multiplicative coefficient to be used on the result in order to ensure k Lipschitzity.
        """
        if self.coef_lip is None:
            raise RuntimeError("compute_coef must be called before calling get_coef")
        return self.coef_lip * self.k_coef_lip


class Condensable:
    """
    Some Layers don't optimize directly the kernel, this means that the kernel stored in the layer is not the kernel
    used to make predictions (called W_bar), to address this, these layers can implement the condense() function that
    make self.kernel equal to W_bar.

    This operation also allow the turn the lipschitz layer to it keras equivalent ie. The Dense layer that have the same
    predictions as the trained SpectralDense.
    """

    def condense(self):
        """
        The condense operation allow to overwrite the kernel and ensure that other variables are still consistent.

        Returns:

        """
        raise NotImplementedError(
            "condense function must be implemented by condenseable layers"
        )

    def vanilla_export(self):
        """
        This operation allow to turn this Layer to it's super type, easing storage and serving.

        Returns: self as super type

        """
        raise NotImplementedError(
            "condense function must be implemented by condenseable layers"
        )


@deel_export
class SpectralDense(Dense, LipschitzLayer, Condensable):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer=BjorckInitializer(
            niter_spectral=DEFAULT_NITER_SPECTRAL_INIT,
            niter_bjorck=DEFAULT_NITER_BJORCK,
        ),
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        k_coef_lip=1.0,
        niter_spectral=DEFAULT_NITER_SPECTRAL,
        niter_bjorck=DEFAULT_NITER_BJORCK,
        **kwargs
    ):
        """
        This class is a Dense Layer constrained such that all singular of it's kernel are 1. The computation based on
        BjorckNormalizer algorithm.
        The computation is done in two steps:

        #. reduce the larget singular value to 1, using iterated power method.
        #. increase other singular values to 1, using BjorckNormalizer algorithm.

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
            niter_spectral: number of iteration to find the maximum singular value.
            niter_bjorck: number of iteration with BjorckNormalizer algorithm.

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
            units,
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
        self._kwargs = kwargs
        self.set_klip_factor(k_coef_lip)
        self.niter_spectral = niter_spectral
        self.niter_bjorck = niter_bjorck
        self.u = None
        self.sig = None
        self.built = False
        if self.niter_spectral < 1:
            raise RuntimeError("niter_spectral has to be > 0")

    def build(self, input_shape):
        super(SpectralDense, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.u = self.add_weight(
            shape=tuple([1, self.kernel.shape.as_list()[-1]]),
            initializer=RandomNormal(0, 1),
            name="sn",
            trainable=False,
        )

        self.sig = self.add_weight(
            shape=tuple([1, 1]),  # maximum spectral  value
            name="sigma",
            trainable=False,
        )
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        return 1.0  # this layer don't require a corrective factor

    @tf.function
    def call(self, x, training=None):
        if training:
            W_bar, _u, sigma = spectral_normalization(
                self.kernel, self.u, niter=self.niter_spectral
            )
            self.sig.assign(sigma)
            self.u.assign(_u)
        else:
            W_reshaped = K.reshape(self.kernel, [-1, self.kernel.shape[-1]])
            W_bar = W_reshaped / self.sig

        W_bar = bjorck_normalization(W_bar, niter=self.niter_bjorck)
        W_bar = W_bar * self._get_coef()

        # with tf.control_dependencies([self.u.assign(_u), self.sig.assign(sigma)]):
        W_bar = K.reshape(W_bar, self.kernel.shape)
        kernel = self.kernel
        self.kernel = W_bar
        outputs = Dense.call(self, x)
        self.kernel = kernel
        return outputs

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "niter_spectral": self.niter_spectral,
            "niter_bjorck": self.niter_bjorck,
        }
        base_config = super(SpectralDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def condense(self):
        W_shape = self.kernel.shape
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        new_w = W_reshaped / self.sig.numpy()
        new_w = bjorck_normalization(new_w, niter=self.niter_bjorck)
        # new_w = new_w * self._get_coef()
        new_w = K.reshape(new_w, W_shape)
        self.kernel.assign(new_w)
        # update the u vector as it was computed for previous kernel
        W_bar, _u, sigma = spectral_normalization(
            self.kernel, self.u, niter=self.niter_spectral
        )
        self.u.assign(_u)
        self.sig.assign(sigma)

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        layer = Dense(
            self.units,
            self.activation,
            self.use_bias,
            "glorot_uniform",
            "zeros",
            None,
            None,
            None,
            None,
            None,
            **self._kwargs
        )
        layer.build(self.input_shape)
        layer.kernel.assign(self.kernel.numpy() * self._get_coef())
        if self.use_bias:
            layer.bias.assign(self.bias.numpy())
        return layer


@deel_export
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
        kernel_initializer=BjorckInitializer(
            niter_spectral=DEFAULT_NITER_SPECTRAL_INIT,
            niter_bjorck=DEFAULT_NITER_BJORCK,
        ),
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        k_coef_lip=1.0,
        niter_spectral=DEFAULT_NITER_SPECTRAL,
        niter_bjorck=DEFAULT_NITER_BJORCK,
        **kwargs
    ):
        """
        This class is a Conv2D Layer constrained such that all singular of it's kernel are 1. The computation based on
        BjorckNormalizer algorithm. As this is not enough to ensure 1 Lipschitzity a coertive coefficient is applied on the
        output.
        The computation is done in three steps:

        #. reduce the largest singular value to 1, using iterated power method.
        #. increase other singular values to 1, using BjorckNormalizer algorithm.
        #. divide the output by the Lipschitz bound to ensure k Lipschitzity.

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
            niter_spectral: number of iteration to find the maximum singular value.
            niter_bjorck: number of iteration with BjorckNormalizer algorithm.

        This documentation reuse the body of the original keras.layers.Conv2D doc.
        """
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
        super(SpectralConv2D, self).__init__(
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
            **kwargs
        )
        self._kwargs = kwargs
        self.set_klip_factor(k_coef_lip)
        self.u = None
        self.sig = None
        self.niter_spectral = niter_spectral
        self.niter_bjorck = niter_bjorck
        if self.niter_spectral < 1:
            raise RuntimeError("niter_spectral has to be > 0")

    def build(self, input_shape):
        super(SpectralConv2D, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.u = self.add_weight(
            shape=tuple([1, self.kernel.shape.as_list()[-1]]),
            initializer=RandomNormal(0, 1),
            name="sn",
            trainable=False,
        )

        self.sig = self.add_weight(
            shape=tuple([1, 1]),  # maximum spectral  value
            name="sigma",
            trainable=False,
        )
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        # According to the file lipschitz_CNN.pdf
        stride = np.prod(self.strides)
        # print(stride)
        # print(inputs.shape)
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
            print("coef before correction : " + str(np.sqrt(k1 * k2)))
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
            ho = np.floor(h / sn1)
            wo = np.floor(w / sn2)
            alpha1 = np.ceil(k1 / sn1)
            alpha2 = np.ceil(k2 / sn2)
            print("coef before correction : " + str(np.sqrt(alpha1 * alpha2)))
            alphabar1 = np.floor(k1_div2 / sn1)
            alphabar2 = np.floor(k2_div2 / sn2)
            betabar1 = k1_div2 - alphabar1 * sn1
            betabar2 = k2_div2 - alphabar2 * sn2
            zl1 = (alphabar1 * sn1 + 2 * betabar1) * (alphabar1 + 1) / 2
            zl2 = (alphabar2 * sn2 + 2 * betabar2) * (alphabar2 + 1) / 2
            gamma1 = h - 1 - sn1 * np.ceil((h - 1 - k1_div2) / sn1)
            gamma2 = w - 1 - sn2 * np.ceil((w - 1 - k2_div2) / sn2)
            alphah1 = np.floor(gamma1 / sn1)
            alphaw2 = np.floor(gamma2 / sn2)
            print("gamma2:  {}, alphaw2 {}".format(gamma2, alphaw2))
            """delta1 = (ho-1-gamma1)*sn1+k1_div2-1-h-k1_div2
            delta2 = (wo-1-gamma2)*sn2+k2_div2-1-w-k2_div2
            zr1 = (gamma1*sn1+2*delta1)*(gamma1+1)/2
            zr2 = (gamma2*sn2+2*delta2)*(gamma2+1)/2"""
            zr1 = (alphah1 + 1) * (k1_div2 - gamma1 + sn1 * alphah1 / 2.0)
            zr2 = (alphaw2 + 1) * (k2_div2 - gamma2 + sn2 * alphaw2 / 2.0)
            print(
                "zeros: up {}, left {}, bottom {}, right {}".format(zl1, zl2, zr1, zr2)
            )
            coefLip = np.sqrt((h * w) / ((k1 * ho - zl1 - zr1) * (k2 * wo - zl2 - zr2)))
        return coefLip

    @tf.function
    def call(self, x, training=None):
        W_shape = self.kernel.shape
        if training:
            W_bar, _u, sigma = spectral_normalization(
                self.kernel, self.u, niter=self.niter_spectral
            )
            self.sig.assign(sigma)
            self.u.assign(_u)
        else:
            W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
            W_bar = W_reshaped / self.sig
        W_bar = bjorck_normalization(W_bar, niter=self.niter_bjorck)
        W_bar = W_bar * self._get_coef()

        W_bar = K.reshape(W_bar, W_shape)
        outputs = K.conv2d(
            x,
            W_bar,
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
            "niter_spectral": self.niter_spectral,
            "niter_bjorck": self.niter_bjorck,
        }
        base_config = super(SpectralConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def condense(self):
        W_shape = self.kernel.shape
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        new_w = W_reshaped / self.sig.numpy()
        new_w = bjorck_normalization(new_w, niter=self.niter_bjorck)
        # new_w = new_w * self._get_coef()
        new_w = K.reshape(new_w, W_shape)
        self.kernel.assign(new_w)
        # update the u vector as it was computed for previous kernel
        W_bar, _u, sigma = spectral_normalization(
            self.kernel, self.u, niter=self.niter_spectral
        )
        self.u.assign(_u)
        self.sig.assign(sigma)

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        layer = Conv2D(
            self.filters,
            self.kernel_size,
            self.strides,
            self.padding,
            self.data_format,
            self.dilation_rate,
            self.activation,
            self.use_bias,
            "glorot_uniform",
            "zeros",
            None,
            None,
            None,
            None,
            None,
            **self._kwargs
        )
        layer.build(self.input_shape)
        layer.kernel.assign(self.kernel.numpy() * self._get_coef())
        if self.use_bias:
            layer.bias.assign(self.bias.numpy())
        return layer


@deel_export
class FrobeniusDense(Dense, LipschitzLayer, Condensable):
    """
    Same a SpectralDense, but in the case of a single output.
    """

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer=SpectralInitializer(
            niter_spectral=DEFAULT_NITER_SPECTRAL_INIT
        ),
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        k_coef_lip=1.0,
        **kwargs
    ):
        super().__init__(
            units,
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
        self.set_klip_factor(k_coef_lip)
        self._kwargs = kwargs

    def build(self, input_shape):
        self._init_lip_coef(input_shape)
        return super(FrobeniusDense, self).build(input_shape)

    def _compute_lip_coef(self, input_shape=None):
        return 1.0

    @tf.function
    def call(self, x):
        W_bar = self.kernel / tf.norm(self.kernel) * self._get_coef()
        kernel = self.kernel
        self.kernel = W_bar
        outputs = Dense.call(self, x)
        self.kernel = kernel
        return outputs

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
        }
        base_config = super(FrobeniusDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def condense(self):
        W_bar = self.kernel / tf.norm(self.kernel)
        self.kernel.assign(W_bar)

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        layer = Dense(
            self.units,
            self.activation,
            self.use_bias,
            "glorot_uniform",
            "zeros",
            None,
            None,
            None,
            None,
            None,
            **self._kwargs
        )
        layer.build(self.input_shape)
        layer.kernel.assign(self.kernel.numpy() * self._get_coef())
        if self.use_bias:
            layer.bias.assign(self.bias.numpy())
        return layer


@deel_export
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
        kernel_initializer=SpectralInitializer(
            niter_spectral=DEFAULT_NITER_SPECTRAL_INIT
        ),
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
            or isinstance(kernel_constraint, BjorckNormalizer)
            or isinstance(kernel_constraint, SpectralNormalizer)
        ):
            raise RuntimeError(
                "only deellip constraints are allowed as constraints could break 1 lipschitz condition"
            )
        super(FrobeniusConv2D, self).__init__(
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
            **kwargs
        )
        self.set_klip_factor(k_coef_lip)
        self._kwargs = kwargs

    def build(self, input_shape):
        self._init_lip_coef(input_shape)
        return super(FrobeniusConv2D, self).build(input_shape)

    def _compute_lip_coef(self, input_shape=None):
        return SpectralConv2D._compute_lip_coef(self, input_shape)

    @tf.function
    def call(self, x):
        W_bar = (self.kernel / tf.norm(self.kernel)) * self._get_coef()
        outputs = K.conv2d(
            x,
            W_bar,
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
        self.kernel.assign(self.kernel / tf.norm(self.kernel))

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        # call the condense function from SpectralDense as if it was from this class
        return SpectralConv2D.vanilla_export(self)


@deel_export
class ScaledAveragePooling2D(AveragePooling2D, LipschitzLayer):
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

        This documentation reuse the body of the original keras.layers.AveragePooling2D doc.
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

    @tf.function
    def call(self, x, training=None):
        return super(AveragePooling2D, self).call(x) * self._get_coef()

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
        }
        base_config = super(ScaledAveragePooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@deel_export
class ScaledMaxPooling2D(KerasMaxPooling2D, LipschitzLayer):
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
        Global Max pooling operation for 3D data. Same as keras.MaxPooling2D but with lipschitz corrective factor.

        Arguments:
            data_format: A string,
                one of `channels_last` (default) or `channels_first`.
                The ordering of the dimensions in the inputs.
                `channels_last` corresponds to inputs with shape
                `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
                while `channels_first` corresponds to inputs with shape
                `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
                It defaults to the `image_data_format` value found in your
                Keras config file at `~/.keras/keras.json`.
                If you never set it, then it will be "channels_last".

        Input shape:
            - If `data_format='channels_last'`:
                5D tensor with shape:
                `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            - If `data_format='channels_first'`:
                5D tensor with shape:
                `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

        Output shape:
            2D tensor with shape `(batch_size, channels)`.

        This documentation reuse the body of the original keras.layers.MaxPooling doc.
        """
        if not ((strides == pool_size) or (strides is None)):
            raise RuntimeError("stride must be equal to pool_size")
        if padding != "valid":
            raise RuntimeError("ScaledMaxPooling only support padding='valid'")
        self.set_klip_factor(k_coef_lip)
        self._kwargs = kwargs
        super().__init__(pool_size, pool_size, padding, data_format, **kwargs)

    def build(self, input_shape):
        super(ScaledMaxPooling2D, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        return 1.0

    @tf.function
    def call(self, x, training=None):
        return super(ScaledMaxPooling2D, self).call(x) * self._get_coef()

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
        }
        base_config = super(ScaledMaxPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@deel_export
class ScaledL2NormPooling2D(AveragePooling2D, LipschitzLayer):

    def __init__(self,
                 pool_size=(2, 2),
                 strides=None,
                 padding='valid',
                 data_format=None,
                 k_coef_lip=1.0,
                 **kwargs):
        """
        Average pooling operation for spatial data, with a lipschitz bound. This pooling operation is norm preserving
        (aka gradient=1 almost everywhere).

        [1]Y.-L.Boureau, J.Ponce, et Y.LeCun, « A Theoretical Analysis of Feature Pooling in Visual Recognition »,p.8.

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
        """
        if not ((strides == pool_size) or (strides is None)):
            raise RuntimeError("stride must be equal to pool_size")
        if padding != "valid":
            raise RuntimeError("NormalizedConv only support padding='valid'")
        super(ScaledL2NormPooling2D, self).__init__(pool_size=pool_size, strides=pool_size, padding=padding,
                                                     data_format=data_format, **kwargs)
        self.set_klip_factor(k_coef_lip)
        self._kwargs = kwargs

    def build(self, input_shape):
        super(AveragePooling2D, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        return np.sqrt(np.prod(np.asarray(self.pool_size)))

    @tf.function
    def call(self, x, training=None):
        return tf.sqrt(super(AveragePooling2D, self).call(tf.square(x))) * self._get_coef()

    def get_config(self):
        config = {
            'k_coef_lip': self.k_coef_lip,
        }
        base_config = super(AveragePooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@deel_export
class ScaledGlobalAveragePooling2D(GlobalAveragePooling2D, LipschitzLayer):
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

        This documentation reuse the body of the original keras.layers.GlobalAveragePooling doc.
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

    @tf.function
    def call(self, x, training=None):
        return super(ScaledGlobalAveragePooling2D, self).call(x) * self._get_coef()

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
        }
        base_config = super(ScaledGlobalAveragePooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
