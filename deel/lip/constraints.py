# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains extra constraint objects. These object can be added as params to
regular layers.
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from .normalizers import (
    reshaped_kernel_orthogonalization,
    DEFAULT_EPS_SPECTRAL,
    DEFAULT_EPS_BJORCK,
    DEFAULT_BETA_BJORCK,
)
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable("deel-lip", "WeightClipConstraint")
class WeightClipConstraint(Constraint):
    def __init__(self, c=2):
        """
        Clips the weights incident to each hidden unit to be inside the range `[-c,+c]`.

        Args:
            c (float): clipping parameter.
        """
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {"c": self.c}


@register_keras_serializable("deel-lip", "AutoWeightClipConstraint")
class AutoWeightClipConstraint(Constraint):
    def __init__(self, scale=1):
        """
        Clips the weights incident to each hidden unit to be inside the range `[-c,+c]`.
        With c = 1/sqrt(size(kernel)).

        Args:
            scale (float): scaling factor to increase/decrease clipping value.
        """
        self.scale = scale

    def __call__(self, w):
        c = 1 / (tf.sqrt(tf.cast(tf.size(w), dtype=w.dtype)) * self.scale)
        return tf.clip_by_value(w, -c, c)

    def get_config(self):
        return {"scale": self.scale}


@register_keras_serializable("deel-lip", "FrobeniusConstraint")
class FrobeniusConstraint(Constraint):
    # todo: duplicate of keras/constraints/UnitNorm ?

    def __init__(self, eps=1e-7):
        """
        Constrain the weights by dividing the weight matrix by it's L2 norm.
        """
        self.eps = eps

    def __call__(self, w):
        return w / (tf.sqrt(tf.reduce_sum(tf.square(w), keepdims=False)) + self.eps)

    def get_config(self):
        return {"eps": self.eps}


@register_keras_serializable("deel-lip", "SpectralConstraint")
class SpectralConstraint(Constraint):
    def __init__(
        self,
        k_coef_lip=1.0,
        eps_spectral=DEFAULT_EPS_SPECTRAL,
        eps_bjorck=DEFAULT_EPS_BJORCK,
        beta_bjorck=DEFAULT_BETA_BJORCK,
        u=None,
    ) -> None:
        """
        Ensure that *all* singular values of the weight matrix equals to 1. Computation
        based on Bjorck algorithm. The computation is done in two steps:

        1. reduce the larget singular value to k_coef_lip, using iterate power method.
        2. increase other singular values to k_coef_lip, using bjorck algorithm.

        Args:
            k_coef_lip (float): lipschitz coefficient of the weight matrix
            eps_spectral (float): stopping criterion for the iterative power algorithm.
            eps_bjorck (float): stopping criterion Bjorck algorithm.
            beta_bjorck (float): beta parameter in bjorck algorithm.
            u (tf.Tensor): vector used for iterated power method, can be set to None
                (used for serialization/deserialization purposes).
        """
        self.eps_spectral = eps_spectral
        self.eps_bjorck = eps_bjorck
        self.beta_bjorck = beta_bjorck
        self.k_coef_lip = k_coef_lip
        if not (isinstance(u, tf.Tensor) or (u is None)):
            u = tf.convert_to_tensor(u)
        self.u = u
        super(SpectralConstraint, self).__init__()

    def __call__(self, w):
        wbar, _, _ = reshaped_kernel_orthogonalization(
            w,
            self.u,
            self.k_coef_lip,
            self.eps_spectral,
            self.eps_bjorck,
            self.beta_bjorck,
        )
        return wbar

    def get_config(self):
        config = {
            "k_coef_lip": self.k_coef_lip,
            "eps_spectral": self.eps_spectral,
            "eps_bjorck": self.eps_bjorck,
            "beta_bjorck": self.beta_bjorck,
            "u": None if self.u is None else self.u.numpy(),
        }
        base_config = super(SpectralConstraint, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
