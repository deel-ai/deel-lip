# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains extra constraint objects. These object can be added as params to regular layers.
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from .normalizers import bjorck_normalization, spectral_normalization
from .utils import _deel_export


@_deel_export
class WeightClip(Constraint):
    def __init__(self, c=2):
        """
        Clips the weights incident to each hidden unit to be inside the range `[-c,+c]`.

        Args:
            c: clipping parameter.
        """
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {"name": self.__class__.__name__, "c": self.c}


@_deel_export
class AutoWeightClip(Constraint):
    def __init__(self, scale=1):
        """
        Clips the weights incident to each hidden unit to be inside the range `[-c,+c]`. With c = 1/sqrt(size(kernel)).

        Args:
            scale: scaling factor to increase/decrease clipping value.
        """
        self.scale = scale
        self.c = None

    def __call__(self, w):
        self.c = 1 / (tf.sqrt(tf.cast(tf.size(w), dtype=tf.float64)) * self.scale)
        return tf.clip_by_value(w, -self.c, self.c)

    def get_config(self):
        return {"name": self.__class__.__name__, "scale": self.scale, "c": self.c}


@_deel_export
class FrobeniusNormalizer(Constraint):
    # todo: duplicate of keras/constraints/UnitNorm ?

    def __init__(self, **kwargs):
        """
        Clips the weights incident to each hidden unit to be inside the range `[-c,+c]`. With c = 1/norm(kernel).
        """
        super(FrobeniusNormalizer, self).__init__(**kwargs)

    def __call__(self, w):
        return w * tf.sqrt(tf.reduce_sum(tf.square(w), keepdims=False))


@_deel_export
class SpectralNormalizer(Constraint):
    def __init__(self, niter_spectral=3, u=None) -> None:
        """
        Ensure that the weights matrix have sigma_max == 1 ( maximum singular value of the weights matrix).

        Args:
            niter_spectral: number of iteration to find the maximum singular value.
            u: vector used for iterated power method, can be set to None ( used for serialization/deserialization
            purposes).
        """
        self.niter_spectral = niter_spectral
        if not (isinstance(u, tf.Tensor) or (u is None)):
            u = tf.convert_to_tensor(u)
        self.u = u
        super(SpectralNormalizer, self).__init__()

    def __call__(self, w):
        w_bar, self.u, sigma = spectral_normalization(
            super(SpectralNormalizer, self).__call__(w),
            self.u,
            niter=self.niter_spectral,
        )
        return K.reshape(w_bar, w.shape)

    def get_config(self):
        config = {
            "niter_spectral": self.niter_spectral,
            "u": None if self.u is None else self.u.numpy(),
        }
        base_config = super(SpectralNormalizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@_deel_export
class BjorckNormalizer(SpectralNormalizer):
    def __init__(self, niter_spectral=3, niter_bjorck=15, u=None) -> None:
        """
        Ensure that *all* singular values of the weight matrix equals to 1. Computation based on BjorckNormalizer algorithm.
        The computation is done in two steps:

        1. reduce the larget singular value to 1, using iterated power method.
        2. increase other singular values to 1, using BjorckNormalizer algorithm.

        Args:
            niter_spectral: number of iteration to find the maximum singular value.
            niter_bjorck: number of iteration with BjorckNormalizer algorithm..
            u: vector used for iterated power method, can be set to None ( used for serialization/deserialization
            purposes).
        """
        self.niter_bjorck = niter_bjorck
        super(BjorckNormalizer, self).__init__(niter_spectral, u)

    def __call__(self, w):
        w_bar, self.u, sigma = spectral_normalization(
            w, self.u, niter=self.niter_spectral
        )
        w_bar = bjorck_normalization(w_bar, niter=self.niter_bjorck)
        return K.reshape(w_bar, shape=w.shape)

    def get_config(self):
        config = {"niter_bjorck": self.niter_bjorck}
        base_config = super(BjorckNormalizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
