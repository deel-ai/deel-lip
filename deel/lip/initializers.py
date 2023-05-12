# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
from tensorflow.keras.initializers import Initializer
from tensorflow.keras import initializers
from .normalizers import (
    reshaped_kernel_orthogonalization,
    DEFAULT_EPS_SPECTRAL,
    DEFAULT_EPS_BJORCK,
    DEFAULT_BETA_BJORCK,
)

from .utils_lorth import (
    Lorth2Dgrad,
    DEFAULT_LAMBDA_LORTHGRAD,
    DEFAULT_NITER_INITIALIZER_LORTHGRAD,
)


from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable("deel-lip", "SpectralInitializer")
class SpectralInitializer(Initializer):
    def __init__(
        self,
        eps_spectral=DEFAULT_EPS_SPECTRAL,
        eps_bjorck=DEFAULT_EPS_BJORCK,
        beta_bjorck=DEFAULT_BETA_BJORCK,
        k_coef_lip=1.0,
        base_initializer="orthogonal",
    ) -> None:
        """
        Initialize a kernel to be 1-lipschitz orthogonal using bjorck
        normalization.

        Args:
            eps_spectral (float): stopping criterion of iterative power method
            eps_bjorck (float): float greater than 0, stopping criterion of
                bjorck algorithm, setting it to None disable orthogonalization
            beta_bjorck (float): beta parameter of bjorck algorithm
            base_initializer (str): method used to generate weights before applying the
                orthonormalization
        """
        self.eps_spectral = eps_spectral
        self.eps_bjorck = eps_bjorck
        self.beta_bjorck = beta_bjorck
        self.k_coef_lip = k_coef_lip
        self.base_initializer = initializers.get(base_initializer)
        super(SpectralInitializer, self).__init__()

    def __call__(self, shape, dtype=None, partition_info=None):
        w = self.base_initializer(shape=shape, dtype=dtype)
        wbar, u, sigma = reshaped_kernel_orthogonalization(
            w,
            None,
            self.k_coef_lip,
            self.eps_spectral,
            self.eps_bjorck,
            self.beta_bjorck,
        )
        return wbar

    def get_config(self):
        return {
            "eps_spectral": self.eps_spectral,
            "eps_bjorck": self.eps_bjorck,
            "beta_bjorck": self.beta_bjorck,
            "k_coef_lip": self.k_coef_lip,
            "base_initializer": initializers.serialize(self.base_initializer),
        }


@register_keras_serializable("deel-lip", "LorthInitializer")
class LorthInitializer(Initializer):
    def __init__(
        self,
        niter_newton=DEFAULT_NITER_INITIALIZER_LORTHGRAD,
        lambda_step=DEFAULT_LAMBDA_LORTHGRAD,
        base_initializer="glorot_uniform",
        k_coef_lip=1.0,
    ) -> None:
        """
        Initialize a kernel to be 1-lipschitz orthogonal using bjorck
        normalization.

        Args:
            niter_newton: iteration steps for newton
            lambda_step: float greater than 0, max step size for newton
            base_initializer: method used to generate weights before applying the
                orthonormalization
        """
        self.niter_newton = niter_newton
        self.lambda_step = lambda_step
        self.k_coef_lip = k_coef_lip
        assert k_coef_lip == 1.0, "Not implemented"
        self.base_initializer = initializers.get(base_initializer)
        super(LorthInitializer, self).__init__()

    def __call__(self, shape, stride=1.0, dtype=None, partition_info=None):
        w = self.base_initializer(shape=shape, dtype=dtype)
        lorthg = Lorth2Dgrad(
            kernel_shape=shape,
            stride=stride,
            niter_newton=self.niter_newton,
            lambda_step=self.lambda_step,
        )

        wbar = lorthg.lorthGradient_orthogonalization(w, verbose=False)
        return wbar

    def get_config(self):
        return {
            "niter_newton": self.niter_newton,
            "lambda_step": self.lambda_step,
            "k_coef_lip": self.k_coef_lip,
            "base_initializer": initializers.serialize(self.base_initializer),
        }
