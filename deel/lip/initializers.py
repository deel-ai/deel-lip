# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
from tensorflow.keras.initializers import Initializer, RandomUniform
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from .normalizers import spectral_normalization, bjorck_normalization
from .utils import _deel_export


@_deel_export
class SpectralInitializer(Initializer):
    def __init__(
        self,
        niter_spectral=3,
        base_initializer=RandomUniform(minval=0.0, maxval=0.5, seed=None),
    ) -> None:
        """
        Initialize a kernel to be 1-lipschitz using spectral normalization (iterative
        power method).

        Args:
            niter_spectral: number of iteration to do with the iterative power method
            base_initializer: method used to generat weights before applying iterative
                power method
        """
        self.niter_spectral = niter_spectral
        self.base_initializer = initializers.get(base_initializer)
        super(SpectralInitializer, self).__init__()

    def __call__(self, shape, dtype=None, partition_info=None):
        w = self.base_initializer(shape=shape, dtype=dtype)
        u = K.random_uniform(shape=tuple([1, shape[-1]]), dtype=dtype)
        w, _u, sigma = spectral_normalization(w, u, self.niter_spectral)
        return K.reshape(w, shape)

    def get_config(self):
        return {
            "niter_spectral": self.niter_spectral,
            "base_initializer": initializers.serialize(self.base_initializer),
        }


@_deel_export
class BjorckInitializer(Initializer):
    def __init__(
        self,
        niter_spectral=3,
        niter_bjorck=15,
        base_initializer=RandomUniform(minval=0.0, maxval=0.5, seed=None),
    ) -> None:
        """
        Initialize a kernel to be 1-lipschitz almost everywhere using bjorck
        normalization.

        Args:
            niter_spectral: number of iteration to do with the iterative power method
            niter_bjorck: number of iteration to do with the bjorck algorithm
            base_initializer: method used to generat weights before applying the
                orthonormalization
        """
        self.niter_spectral = niter_spectral
        self.niter_bjorck = niter_bjorck
        self.base_initializer = initializers.get(base_initializer)
        super(BjorckInitializer, self).__init__()

    def __call__(self, shape, dtype=None, partition_info=None):
        w = self.base_initializer(shape=shape, dtype=dtype)
        u = K.random_uniform(shape=tuple([1, shape[-1]]), dtype=dtype)
        w_bar, _u, sigma = spectral_normalization(w, u, self.niter_spectral)
        w_bar = bjorck_normalization(w_bar, self.niter_bjorck)
        return K.reshape(w_bar, shape)

    def get_config(self):
        return {
            "niter_spectral": self.niter_spectral,
            "niter_bjorck": self.niter_bjorck,
            "base_initializer": initializers.serialize(self.base_initializer),
        }
