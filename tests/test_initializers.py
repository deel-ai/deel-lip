# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K, metrics
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from deel.lip.initializers import SpectralInitializer

FIT = "fit_generator" if tf.__version__.startswith("2.0") else "fit"
EVALUATE = "evaluate_generator" if tf.__version__.startswith("2.0") else "evaluate"

np.random.seed(42)


class MyTestCase(unittest.TestCase):
    def test_bjorck_initializer(self):
        input_shape = (5,)
        model = Sequential([Dense(4, kernel_initializer=SpectralInitializer(30, 15))])
        self._test_model(model, input_shape)
        model = Sequential([Dense(100, kernel_initializer=SpectralInitializer(30, 15))])
        self._test_model(model, input_shape)

    def _test_model(self, model, input_shape):
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
        np.testing.assert_allclose(sigmas, np.ones_like(sigmas), 1e-6, 0)


if __name__ == "__main__":
    unittest.main()
