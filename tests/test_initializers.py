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
from tests.test_layers import linear_generator, build_kernel
from deel.lip.initializers import SpectralInitializer, BjorckInitializer
from deel.lip.utils import evaluate_lip_const

FIT = "fit_generator" if tf.__version__.startswith("2.0") else "fit"
EVALUATE = "evaluate_generator" if tf.__version__.startswith("2.0") else "evaluate"

np.random.seed(42)


class MyTestCase(unittest.TestCase):
    def test_spectral_initializer(self):
        input_shape = (5,)
        model = Sequential([Dense(4, kernel_initializer=SpectralInitializer(3))])
        self._test_model(model, input_shape)

    def test_bjorck_initializer(self):
        input_shape = (5,)
        model = Sequential([Dense(4, kernel_initializer=BjorckInitializer(3, 15))])
        self._test_model(model, input_shape)

    def _test_model(self, model, input_shape):
        batch_size = 1000
        k_lip_data = 1.0
        # clear session to avoid side effects from previous train
        K.clear_session()
        # create the keras model, defin opt, and compile it
        optimizer = Adam(lr=0.001)
        model.compile(
            optimizer=optimizer, loss="mean_squared_error", metrics=[metrics.mse]
        )
        # model.summary()
        # create the synthetic data generator
        output_shape = model.compute_output_shape((batch_size,) + input_shape)[1:]
        kernel = build_kernel(input_shape, output_shape, k_lip_data)
        # define logging features
        # the seed is set to compare all models with the same data
        x, y = linear_generator(batch_size, input_shape, kernel).send(None)
        np.random.seed(42)
        empirical_lip_const = evaluate_lip_const(model=model, x=x, seed=42)
        self.assertLessEqual(
            abs(empirical_lip_const - 1.0),
            0.02,
            msg="the initializer must lead to a NN with Lipschitz constant equal to 1",
        )


if __name__ == "__main__":
    unittest.main()
