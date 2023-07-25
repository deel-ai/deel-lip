# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import unittest
from functools import partial

import numpy as np
import tensorflow as tf
from deel.lip.normalizers import (
    bjorck_normalization,
    reshaped_kernel_orthogonalization,
    spectral_normalization,
    spectral_normalization_conv,
)
from deel.lip.utils import _padding_circular

rng = np.random.default_rng(42)


class TestSpectralNorm(unittest.TestCase):
    """Test of spectral normalization (power iteration) on Dense and Conv2D kernels."""

    def test_spectral_normalization(self):
        # Dense kernel
        kernel_shape = (15, 32)
        kernel = rng.normal(size=kernel_shape).astype("float32")
        self._test_kernel(kernel)
        # Dense kernel projection
        kernel_shape = (32, 15)
        kernel = rng.normal(size=kernel_shape).astype("float32")
        self._test_kernel(kernel)

    def _test_kernel(self, kernel):
        """Compare max singular value using power iteration and tf.linalg.svd"""
        sigmas_svd = tf.linalg.svd(
            np.reshape(kernel, (np.prod(kernel.shape[:-1]), kernel.shape[-1])),
            full_matrices=False,
            compute_uv=False,
        ).numpy()
        SVmax = np.max(sigmas_svd)

        u = rng.normal(size=(1, kernel.shape[-1]))
        W_bar, _u, sigma = spectral_normalization(kernel, u=u, eps=1e-6)
        # Test sigma is close to the one computed with svd first run @ 1e-1
        np.testing.assert_approx_equal(
            sigma, SVmax, 1, "test failed with kernel_shape " + str(kernel.shape)
        )

        W_bar, _u, sigma = spectral_normalization(kernel, u=_u, eps=1e-6)
        W_bar, _u, sigma = spectral_normalization(kernel, u=_u, eps=1e-6)
        # Test W_bar is reshaped correctly
        np.testing.assert_equal(
            W_bar.shape, (np.prod(kernel.shape[:-1]), kernel.shape[-1])
        )
        # Test sigma is close to the one computed with svd second run @ 1e-5
        np.testing.assert_approx_equal(
            sigma, SVmax, 3, "test failed with kernel_shape " + str(kernel.shape)
        )
        # Test if kernel is normalized by sigma
        np.testing.assert_allclose(
            np.reshape(W_bar, kernel.shape), kernel / sigmas_svd[0], 1e-2, 0
        )


class TestSpectralNormConv(unittest.TestCase):
    """Test of conv spectral normalization (power iteration conv) on Conv2D kernels.

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
        kernel_shape = (3, 3, 24, 24)
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self.strides = [1, 1]
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
        """Compare power iteration conv against SVD."""

        # Compute max singular value using FFT2 and SVD
        kernel_n = kernel.astype(dtype="float32")
        transforms = np.fft.fft2(
            kernel_n,
            (self.spectral_input_shape[0], self.spectral_input_shape[1]),
            axes=[0, 1],
        )
        svd = np.linalg.svd(transforms, compute_uv=False)
        SVmax = np.max(svd)

        # Compute max singular value using power iteration conv
        _u = np.random.normal(size=(1,) + self.spectral_input_shape).astype("float32")
        fPad = partial(_padding_circular, circular_paddings=self.cPad)

        W_bar, _u, sigma = spectral_normalization_conv(
            kernel,
            u=_u,
            stride=self.strides[0],
            conv_first=not self.RO_case,
            pad_func=fPad,
            eps=1e-6,
            maxiter=30,
        )
        # Test if sigma is close to the one computed with svd first run @ 1e-1
        np.testing.assert_approx_equal(
            sigma, SVmax, 1, "test failed with kernel_shape " + str(kernel.shape)
        )

        # Run a second time power iteration conv with last _u from first run
        W_bar, _u, sigma = spectral_normalization_conv(
            kernel,
            u=_u,
            stride=self.strides[0],
            conv_first=not self.RO_case,
            pad_func=fPad,
            eps=1e-6,
            maxiter=30,
        )
        # Test if W_bar is reshaped correctly
        np.testing.assert_equal(W_bar.shape, kernel.shape)
        # Test if sigma is close to the one computed with svd, second run
        np.testing.assert_approx_equal(
            sigma, SVmax, 2, "test failed with kernel_shape " + str(kernel.shape)
        )
        # Test if kernel is normalized by sigma
        np.testing.assert_allclose(
            np.reshape(W_bar, kernel.shape), kernel / SVmax, 1e-2, 0
        )


class TestBjorckNormalization(unittest.TestCase):
    """Test of Björck orthogonalization on Dense and Conv2D kernels."""

    def test_bjorck_normalization(self):
        # Dense kernel
        kernel_shape = (15, 32)
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self._test_kernel(kernel)

        # Conv kernel
        kernel_shape = (64, 32)
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self._test_kernel(kernel)

    def _test_kernel(self, kernel):
        """Compare max singular value using power iteration and tf.linalg.svd"""
        sigmas_svd = tf.linalg.svd(
            np.reshape(kernel, (np.prod(kernel.shape[:-1]), kernel.shape[-1])),
            full_matrices=False,
            compute_uv=False,
        ).numpy()
        SVmax = np.max(sigmas_svd)

        wbar = bjorck_normalization(kernel / SVmax, eps=1e-5)
        sigmas_wbar_svd = tf.linalg.svd(
            np.reshape(wbar, (np.prod(wbar.shape[:-1]), wbar.shape[-1])),
            full_matrices=False,
            compute_uv=False,
        ).numpy()
        # Test sigma is close to the one computed with svd first run @ 1e-1
        np.testing.assert_allclose(
            sigmas_wbar_svd, np.ones(sigmas_wbar_svd.shape), 1e-2, 0
        )
        # Test W_bar is reshaped correctly
        np.testing.assert_equal(
            wbar.shape, (np.prod(kernel.shape[:-1]), kernel.shape[-1])
        )

        # Test sigma is close to the one computed with svd second run @1e-5
        wbar = bjorck_normalization(wbar, eps=1e-5)
        sigmas_wbar_svd = tf.linalg.svd(
            np.reshape(wbar, (np.prod(wbar.shape[:-1]), wbar.shape[-1])),
            full_matrices=False,
            compute_uv=False,
        ).numpy()
        np.testing.assert_allclose(
            sigmas_wbar_svd, np.ones(sigmas_wbar_svd.shape), 1e-4, 0
        )


class TestRKO(unittest.TestCase):
    """Test of RKO algorithm on Dense and Conv2D kernels."""

    def test_reshaped_kernel_orthogonalization(self):
        # Dense kernel
        kernel_shape = (15, 32)
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self._test_kernel(kernel)

        # Conv kernel
        kernel_shape = (5, 5, 64, 32)
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self._test_kernel(kernel)

    def _test_kernel(self, kernel):
        """Compare max singular value using power iteration and tf.linalg.svd"""
        sigmas_svd = tf.linalg.svd(
            np.reshape(kernel, (np.prod(kernel.shape[:-1]), kernel.shape[-1])),
            full_matrices=False,
            compute_uv=False,
        ).numpy()
        SVmax = np.max(sigmas_svd)

        W_bar, _, sigma = reshaped_kernel_orthogonalization(
            kernel, u=None, adjustment_coef=1.0, eps_spectral=1e-5, eps_bjorck=1e-5
        )
        # Test W_bar is reshaped correctly
        np.testing.assert_equal(W_bar.shape, kernel.shape)
        # Test RKO sigma is close to max(svd)
        np.testing.assert_approx_equal(
            sigma, SVmax, 1, "test failed with kernel_shape " + str(kernel.shape)
        )
        sigmas_wbar_svd = tf.linalg.svd(
            np.reshape(W_bar, (np.prod(W_bar.shape[:-1]), W_bar.shape[-1])),
            full_matrices=False,
            compute_uv=False,
        ).numpy()
        # Test if SVs of W_bar are close to one
        np.testing.assert_allclose(
            sigmas_wbar_svd, np.ones(sigmas_wbar_svd.shape), 1e-2, 0
        )
