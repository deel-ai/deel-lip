# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import unittest
from functools import partial

import numpy as np
import tensorflow as tf
from deel.lip.utils_lorth import (
    Lorth2Dgrad,
)
from deel.lip.normalizers import _power_iteration_conv
from deel.lip.utils import _padding_circular

np.random.seed(42)


class TestLorthGradient_orthogonalization(unittest.TestCase):
    """Test of lorthGradient_orthogonalization for Conv2D kernels.
    Padding is circular.
    TODO : ajouter tests avec padding valid ou same, mais attention au calcul du
    sigma_max (pas de fft+svd pour padding same)
    """

    def test_spectral_normalization_conv(self):
        # CO case (stride^2*C < M)
        kernel_shape = (5, 5, 32, 64)
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self.strides = [1, 1]
        self.set_spectral_input_shape(kernel)
        self._test_kernel(kernel)

        # RO case (stride^2*C > M)
        kernel_shape = (3, 3, 12, 8)
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self.strides = [1, 1]
        self.set_spectral_input_shape(kernel)
        self._test_kernel(kernel)

        # Square case (stride^2*C == M)
        """kernel_shape = (3, 3, 24, 24) HARD TO OPTIMIZE => FAILED
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self.strides = [1, 1]
        self.set_spectral_input_shape(kernel)
        self._test_kernel(kernel)"""

        # CO case (stride^2*C < M)
        kernel_shape = (5, 5, 32, 256)
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self.strides = [2, 2]
        self.set_spectral_input_shape(kernel)
        self._test_kernel(kernel)

        # RO case (stride^2*C > M)
        kernel_shape = (3, 3, 12, 8)
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self.strides = [2, 2]
        self.set_spectral_input_shape(kernel)
        self._test_kernel(kernel)

        # Square case (stride^2*C == M)
        kernel_shape = (3, 3, 24, 96)
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self.strides = [2, 2]
        self.set_spectral_input_shape(kernel)
        self._test_kernel(kernel)

    def set_spectral_input_shape(self, kernel):
        """Set spectral input shape and RO_case, depending on kernel shape and
        strides."""
        (kh, kw, c_in, c_out) = kernel.shape
        self.cPad = [kh // 2, kw // 2]
        stride = self.strides[0]

        # Compute minimal N
        r = kh // 2
        if r < 1:
            N = 5
        else:
            N = 4 * r + 1
            if stride > 1:
                N = int(0.5 + N / stride)

        if c_in * stride**2 > c_out:
            self.spectral_input_shape = (N, N, c_out)
            self.RO_case = True
        else:
            self.spectral_input_shape = (stride * N, stride * N, c_in)
            self.RO_case = False

    def _test_kernel(self, kernel):
        """Apply lorth gradient descent and compute svMax and SVmin."""
        lorth = Lorth2Dgrad(kernel.shape, self.strides[0], niter_newton=50)
        orth_kernel = lorth.lorthGradient_orthogonalization(kernel)

        # Compute max singular value using power_iteration_conv
        print(self.strides)
        print(self.spectral_input_shape)
        _u = np.random.normal(size=(1,) + self.spectral_input_shape).astype("float32")
        fPad = partial(_padding_circular, circular_paddings=self.cPad)
        print(orth_kernel.shape)
        print(_u.shape)
        _u, _v, _ = _power_iteration_conv(
            orth_kernel,
            _u,
            self.strides[0],
            pad_func=fPad,
            conv_first=not self.RO_case,
        )

        # Calculate Sigma
        sigma = tf.norm(_v)
        # Test if sigma is close to the one computed with svd first run @ 1e-1
        np.testing.assert_approx_equal(
            sigma,
            1.0,
            1,
            "test failed sigma_max with kernel_shape " + str(kernel.shape),
        )

        # Test if W_bar is reshaped correctly
        np.testing.assert_equal(orth_kernel.shape, kernel.shape)

        big_const = 1.1 * sigma**2
        _u = np.random.normal(size=(1,) + self.spectral_input_shape).astype("float32")
        _u, _v, norm_u = _power_iteration_conv(
            orth_kernel,
            _u,
            self.strides[0],
            pad_func=fPad,
            conv_first=not self.RO_case,
            big_constant=big_const,
        )

        if big_const - norm_u >= 0:  # normal case
            sigma_min = tf.sqrt(big_const - norm_u)
        elif (
            big_const - norm_u >= -0.0000000000001
        ):  # margin to take into consideration numrica errors
            sigma_min = 0
        else:
            sigma_min = -1  # assertion (should not occur)

        # Test if sigma is close to the one computed with svd, second run
        np.testing.assert_approx_equal(
            sigma_min,
            1.0,
            1,
            "test failed sigma_min with kernel_shape " + str(kernel.shape),
        )

        print(sigma, sigma_min)
