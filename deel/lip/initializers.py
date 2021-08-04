# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
from tensorflow.keras.initializers import Initializer, Orthogonal
from tensorflow.keras import initializers
from .normalizers import project_kernel
from .utils import _deel_export


@_deel_export
class SpectralInitializer(Initializer):
    def __init__(
        self,
        niter_spectral=3,
        niter_bjorck=15,
        k_coef_lip=1.0,
        base_initializer=Orthogonal(gain=1.0, seed=None),
    ) -> None:
        """
        Initialize a kernel to be 1-lipschitz orthogonal using bjorck
        normalization.

        Args:
            niter_spectral: number of iteration to do with the iterative power method
            niter_bjorck: number of iteration to do with the bjorck algorithm
            base_initializer: method used to generate weights before applying the
                orthonormalization
        """
        self.niter_spectral = niter_spectral
        self.niter_bjorck = niter_bjorck
        self.k_coef_lip = k_coef_lip
        self.base_initializer = initializers.get(base_initializer)
        super(SpectralInitializer, self).__init__()

    def __call__(self, shape, dtype=None, partition_info=None):
        w = self.base_initializer(shape=shape, dtype=dtype)
        wbar, u, sigma = project_kernel(
            w,
            None,
            self.k_coef_lip,
            self.niter_spectral,
            self.niter_bjorck,
        )
        return wbar

    def get_config(self):
        return {
            "niter_spectral": self.niter_spectral,
            "niter_bjorck": self.niter_bjorck,
            "k_coef_lip": self.k_coef_lip,
            "base_initializer": initializers.serialize(self.base_initializer),
        }
