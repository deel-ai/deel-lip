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

import keras
import keras.ops as K
from keras.initializers import RandomNormal
from keras.layers import Dense
from keras.saving import register_keras_serializable

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


@register_keras_serializable("deel-lip", "SpectralDense")
class SpectralDense(Dense, LipschitzLayer, Condensable):
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
        lora_rank=None,
        k_coef_lip=1.0,
        eps_spectral=DEFAULT_EPS_SPECTRAL,
        eps_bjorck=DEFAULT_EPS_BJORCK,
        beta_bjorck=DEFAULT_BETA_BJORCK,
        maxiter_spectral=DEFAULT_MAXITER_SPECTRAL,
        maxiter_bjorck=DEFAULT_MAXITER_BJORCK,
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
            lora_rank: SpectralDense only support lora_rank=None.
            k_coef_lip: lipschitz constant to ensure
            eps_spectral: stopping criterion for the iterative power algorithm.
            eps_bjorck: stopping criterion Bjorck algorithm.
            beta_bjorck: beta parameter in bjorck algorithm.
            maxiter_spectral: maximum number of iterations for the power iteration.
            maxiter_bjorck: maximum number of iterations for bjorck algorithm.

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
        if lora_rank is not None:
            raise ValueError("lora_rank is not supported for SpectralDense")

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
            lora_rank=None,
            **kwargs
        )
        self._kwargs = kwargs
        self.set_klip_factor(k_coef_lip)
        _check_RKO_params(eps_spectral, eps_bjorck, beta_bjorck)
        self.eps_spectral = eps_spectral
        self.eps_bjorck = eps_bjorck
        self.beta_bjorck = beta_bjorck
        self.maxiter_bjorck = maxiter_bjorck
        self.maxiter_spectral = maxiter_spectral
        self.u = None
        self.sig = None
        self.wbar = None
        self.built = False

    def build(self, input_shape):
        super(SpectralDense, self).build(input_shape)
        self._init_lip_coef(input_shape)
        self.u = self.add_weight(
            shape=(1, self.units),
            initializer=RandomNormal(0, 1),
            name="sn",
            trainable=False,
            dtype=self.dtype,
        )
        self.sig = self.add_weight(
            shape=(1, 1),  # maximum spectral  value
            initializer=keras.initializers.ones,
            name="sigma",
            trainable=False,
            dtype=self.dtype,
        )
        self.sig.assign([[1.0]])
        self.wbar = keras.Variable(self.kernel.value, trainable=False, name="wbar")
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        return 1.0  # this layer doesn't require a corrective factor

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

        # Compute the output of the Dense layer (copied from keras.layers.Dense)
        x = K.matmul(x, wbar)
        if self.bias is not None:
            x = K.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "eps_spectral": self.eps_spectral,
            "eps_bjorck": self.eps_bjorck,
            "beta_bjorck": self.beta_bjorck,
            "maxiter_spectral": self.maxiter_spectral,
            "maxiter_bjorck": self.maxiter_bjorck,
        }
        base_config = super(SpectralDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_own_variables(self, store):
        super().save_own_variables(store)
        store["sn"] = self.u.numpy()
        store["sigma"] = self.sig.numpy()
        store["wbar"] = self.wbar.numpy()

    def load_own_variables(self, store):
        super().load_own_variables(store)
        self.u.assign(store["sn"])
        self.sig.assign(store["sigma"])
        self.wbar.assign(store["wbar"])

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
        layer = Dense(
            units=self.units,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            **self._kwargs
        )
        layer.build(self.input.shape)
        layer.kernel.assign(self.wbar)
        if self.use_bias:
            layer.bias.assign(self.bias)
        return layer


@register_keras_serializable("deel-lip", "FrobeniusDense")
class FrobeniusDense(Dense, LipschitzLayer, Condensable):
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
        lora_rank=None,
        disjoint_neurons=True,
        k_coef_lip=1.0,
        **kwargs
    ):
        if lora_rank is not None:
            raise ValueError("lora_rank is not supported for FrobeniusDense")

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
            lora_rank=None,
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
        self.wbar = keras.Variable(self.kernel.value, trainable=False, name="wbar")
        self.built = True

    def _compute_lip_coef(self, input_shape=None):
        return 1.0

    def call(self, x, training=True):
        if training:
            wbar = (
                self.kernel
                / K.norm(self.kernel, axis=self.axis_norm)
                * self._get_coef()
            )
            self.wbar.assign(wbar)
        else:
            wbar = self.wbar

        # Compute the output of the Dense layer (copied from keras.layers.Dense)
        x = K.matmul(x, wbar)
        if self.bias is not None:
            x = K.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "disjoint_neurons": self.disjoint_neurons,
        }
        base_config = super(FrobeniusDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def save_own_variables(self, store):
        super().save_own_variables(store)
        store["wbar"] = self.wbar.numpy()

    def load_own_variables(self, store):
        super().load_own_variables(store)
        self.wbar.assign(store["wbar"])

    def condense(self):
        wbar = self.kernel / K.norm(self.kernel, axis=self.axis_norm) * self._get_coef()
        self.kernel.assign(wbar)

    def vanilla_export(self):
        self._kwargs["name"] = self.name
        layer = Dense(
            units=self.units,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            **self._kwargs
        )
        layer.build(self.input.shape)
        layer.kernel.assign(self.wbar)
        if self.use_bias:
            layer.bias.assign(self.bias)
        return layer
