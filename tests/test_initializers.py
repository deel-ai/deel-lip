# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import unittest
import numpy as np
import keras
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from deel.lip.initializers import SpectralInitializer

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
        keras.utils.clear_session()
        # create the keras model, defin opt, and compile it
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss="mean_squared_error",
            metrics=["mean_squared_error"],
        )
        model.build((batch_size,) + input_shape)
        sigmas = np.linalg.svd(
            model.layers[0].kernel,
            full_matrices=False,
            compute_uv=False,
        )
        if orthogonal_test:
            np.testing.assert_allclose(sigmas, np.ones_like(sigmas), 1e-5, 0)
        else:
            np.testing.assert_almost_equal(sigmas.max(), 1.0, 5)


if __name__ == "__main__":
    unittest.main()
