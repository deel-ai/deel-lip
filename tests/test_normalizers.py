# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import unittest

import numpy as np
import tensorflow as tf
from deel.lip.normalizers import (
    bjorck_normalization,
    reshaped_kernel_orthogonalization,
    spectral_normalization,
)

np.random.seed(42)


class TestSpectralNorm(unittest.TestCase):
    """Test of spectral normalization (power iteration) on Dense and Conv2D kernels."""

    def test_spectral_normalization(self):
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

        W_bar, _u, sigma = spectral_normalization(kernel, u=None, eps=1e-6)
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
