# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import unittest
import numpy as np
import tensorflow as tf
from functools import partial
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K, metrics
from tensorflow.keras.layers import Dense, Conv2D, Input
from tensorflow.keras.optimizers import Adam
from deel.lip.initializers import SpectralInitializer, LorthInitializer
from deel.lip.normalizers import _power_iteration_conv
from deel.lip.utils import _padding_circular

FIT = "fit_generator" if tf.__version__.startswith("2.0") else "fit"
EVALUATE = "evaluate_generator" if tf.__version__.startswith("2.0") else "evaluate"

np.random.seed(42)


class MyTestCase(unittest.TestCase):
    def test_bjorck_initializer(self):
        input_shape = (5,)
        model = Sequential(
            [Dense(4, kernel_initializer=SpectralInitializer(1e-6, 1e-6))]
        )
        self._test_model(model, input_shape, orthogonal_test=True)
        model = Sequential(
            [Dense(100, kernel_initializer=SpectralInitializer(1e-6, None))]
        )
        self._test_model(model, input_shape, orthogonal_test=False)

    def _test_model(self, model, input_shape, orthogonal_test=True):
        batch_size = 1000
        # clear session to avoid side effects from previous train
        K.clear_session()
        # create the keras model, defin opt, and compile it
        optimizer = Adam(lr=0.001)
        model.compile(
            optimizer=optimizer, loss="mean_squared_error", metrics=[metrics.mse]
        )
        model.build((batch_size,) + input_shape)
        sigmas = tf.linalg.svd(
            model.layers[0].kernel,
            full_matrices=False,
            compute_uv=False,
        ).numpy()
        if orthogonal_test:
            np.testing.assert_allclose(sigmas, np.ones_like(sigmas), 1e-5, 0)
        else:
            np.testing.assert_almost_equal(sigmas.max(), 1.0, 5)

    def test_lorth_initializer(self):
        # CO case (stride^2*C < M)
        self.kernel_shape = (5, 5, 32, 64)
        self.strides = [1, 1]
        self.set_spectral_input_shape()
        self._test_kernel()

        # RO case (stride^2*C > M)
        self.kernel_shape = (3, 3, 12, 8)
        self.strides = [1, 1]
        self.set_spectral_input_shape()
        self._test_kernel()

        # Square case (stride^2*C == M)
        """kernel_shape = (3, 3, 24, 24) HARD TO OPTIMIZE => FAILED
        kernel = np.random.normal(size=kernel_shape).astype("float32")
        self.strides = [1, 1]
        self.set_spectral_input_shape(kernel)
        self._test_kernel(kernel)"""

        """ERROR no stride in intializer 
        TOD O 
        # CO case (stride^2*C < M)
        self.kernel_shape = (5, 5, 32, 256)
        self.strides = [2, 2]
        self.set_spectral_input_shape()
        self._test_kernel()

        # RO case (stride^2*C > M)
        self.kernel_shape = (3, 3, 12, 8)
        self.strides = [2, 2]
        self.set_spectral_input_shape()
        self._test_kernel()

        # Square case (stride^2*C == M)
        self.kernel_shape = (3, 3, 24, 96)
        self.strides = [2, 2]
        self.set_spectral_input_shape()
        self._test_kernel()
        """

    def set_spectral_input_shape(self):
        """Set spectral input shape and RO_case, depending on kernel shape and
        strides."""
        (kh, kw, c_in, c_out) = self.kernel_shape
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

    def _test_kernel(self):
        """Apply lorth gradient descent and compute svMax and SVmin."""
        input_shape = (5,)
        model = Sequential(
            [
                Input(shape=(1, 64, 64, self.kernel_shape[-2])),
                Conv2D(
                    self.kernel_shape[-1],
                    kernel_size=self.kernel_shape[:2],
                    kernel_initializer=LorthInitializer,
                    strides=self.strides,
                ),
            ]
        )

        orth_kernel = model.layers[-1].kernel

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
            "test failed sigma_max with kernel_shape " + str(self.kernel_shape),
        )

        # Test if W_bar is reshaped correctly
        np.testing.assert_equal(orth_kernel.shape, self.kernel_shape)

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
            "test failed sigma_min with kernel_shape " + str(self.kernel_shape),
        )

        print(sigma, sigma_min)


if __name__ == "__main__":
    unittest.main()
