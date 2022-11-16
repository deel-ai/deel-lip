# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import unittest
from functools import partial

import numpy as np
import tensorflow as tf
from deel.lip.normalizers import (
    spectral_normalization,
    spectral_normalization_conv,
    bjorck_normalization,
    reshaped_kernel_orthogonalization,
)
from deel.lip.utils import _padding_circular

FIT = "fit_generator" if tf.__version__.startswith("2.0") else "fit"
EVALUATE = "evaluate_generator" if tf.__version__.startswith("2.0") else "evaluate"

np.random.seed(42)


class TestSpectralNorm(unittest.TestCase):
    def test_spectral_normalization(self):
        kernel_shape = (15, 32)  # dense type
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self._test_kernel(kernel)
        kernel_shape = (5, 5, 64, 32)  # conv type
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self._test_kernel(kernel)

    def _test_kernel(self, kernel):
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
        # Test sigma is close to the one computed with svd second run @1e-5
        np.testing.assert_approx_equal(
            sigma, SVmax, 3, "test failed with kernel_shape " + str(kernel.shape)
        )
        # Test is kernel is normalized by sigma
        np.testing.assert_allclose(
            np.reshape(W_bar, kernel.shape), kernel / sigmas_svd[0], 1e-2, 0
        )


class TestSpectralNormConv(unittest.TestCase):
    def test_spectral_normalization_conv(self):
        kernel_shape = (5, 5, 32, 64)  # conv type
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self.strides = [1, 1]
        self.set_spectral_input_shape(kernel)
        self._test_kernel(kernel)

        kernel_shape = (3, 3, 12, 8)  # conv type
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self.strides = [1, 1]
        self.set_spectral_input_shape(kernel)
        self._test_kernel(kernel)

        kernel_shape = (3, 3, 24, 24)  # conv type
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self.strides = [1, 1]
        self.set_spectral_input_shape(kernel)
        self._test_kernel(kernel)

    def set_spectral_input_shape(self, kernel):
        (R0, R, C, M) = kernel.shape
        self.cPad = [int(R0 / 2), int(R / 2)]
        stride = self.strides[0]

        # Compute minimal N
        r = R // 2
        if r < 1:
            N = 5
        else:
            N = 4 * r + 1
            if stride > 1:
                N = int(0.5 + N / stride)

        if C * stride**2 > M:
            self.spectral_input_shape = (N, N, M)
            self.RO_case = True
        else:
            self.spectral_input_shape = (stride * N, stride * N, C)
            self.RO_case = False

    def _test_kernel(self, kernel):

        kernel_n = kernel.astype(dtype="float32")
        transforms = np.fft.fft2(
            kernel_n,
            (self.spectral_input_shape[0], self.spectral_input_shape[1]),
            axes=[0, 1],
        )
        svd = np.linalg.svd(transforms, compute_uv=False)
        SVmax = np.max(svd)

        _u = np.random.normal(size=(1,) + self.spectral_input_shape).astype("float32")
        fPad = partial(_padding_circular, circular_paddings=self.cPad)

        W_bar, _u, sigma = spectral_normalization_conv(
            kernel,
            u=_u,
            stride=self.strides[0],
            conv_first=not self.RO_case,
            pad_func=fPad,
            eps=1e-6,
        )
        # Test sigma is close to the one computed with svd first run @ 1e-1
        np.testing.assert_approx_equal(
            sigma, SVmax, 1, "test failed with kernel_shape " + str(kernel.shape)
        )
        W_bar, _u, sigma = spectral_normalization_conv(
            kernel,
            u=_u,
            stride=self.strides[0],
            conv_first=not self.RO_case,
            pad_func=fPad,
            eps=1e-6,
        )
        # Test W_bar is reshaped correctly
        np.testing.assert_equal(W_bar.shape, kernel.shape)
        # Test sigma is close to the one computed with svd second run @1e-5
        np.testing.assert_approx_equal(
            sigma, SVmax, 2, "test failed with kernel_shape " + str(kernel.shape)
        )
        # Test is kernel is normalized by sigma
        np.testing.assert_allclose(
            np.reshape(W_bar, kernel.shape), kernel / SVmax, 1e-2, 0
        )


class TestBjorckNormalization(unittest.TestCase):
    def test_bjorck_normalization(self):
        kernel_shape = (15, 32)  # dense type
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self._test_kernel(kernel)
        kernel_shape = (64, 32)  # conv type
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self._test_kernel(kernel)

    def _test_kernel(self, kernel):
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
    def test_reshaped_kernel_orthogonalization(self):
        kernel_shape = (15, 32)  # dense type
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self._test_kernel(kernel)
        kernel_shape = (5, 5, 64, 32)  # conv type
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self._test_kernel(kernel)

    def _test_kernel(self, kernel):
        sigmas_svd = tf.linalg.svd(
            np.reshape(kernel, (np.prod(kernel.shape[:-1]), kernel.shape[-1])),
            full_matrices=False,
            compute_uv=False,
        ).numpy()
        SVmax = np.max(sigmas_svd)
        W_bar, _u, sigma = reshaped_kernel_orthogonalization(
            kernel, u=None, adjustment_coef=1.0, eps_spectral=1e-5, eps_bjorck=1e-5
        )
        np.testing.assert_approx_equal(
            sigma, SVmax, 1, "test failed with kernel_shape " + str(kernel.shape)
        )
        sigmas_wbar_svd = tf.linalg.svd(
            np.reshape(W_bar, (np.prod(W_bar.shape[:-1]), W_bar.shape[-1])),
            full_matrices=False,
            compute_uv=False,
        ).numpy()
        # Test sigma is close to the one computed with svd first run @ 1e-1
        np.testing.assert_allclose(
            sigmas_wbar_svd, np.ones(sigmas_wbar_svd.shape), 1e-2, 0
        )
        # Test W_bar is reshaped correctly
        np.testing.assert_equal(W_bar.shape, kernel.shape)


if __name__ == "__main__":

    unittest.main()
