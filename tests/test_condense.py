# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import os
import pprint
import unittest
import numpy as np
from tests.test_layers import linear_generator, build_kernel
from deel.lip.model import Sequential, Model
from deel.lip.layers import (
    SpectralDense,
    SpectralConv2D,
    SpectralConv2DTranspose,
    FrobeniusDense,
    FrobeniusConv2D,
    ScaledL2NormPooling2D,
)
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K, Input, metrics, callbacks

FIT = "fit_generator" if tf.__version__.startswith("2.0") else "fit"
EVALUATE = "evaluate_generator" if tf.__version__.startswith("2.0") else "evaluate"

np.random.seed(42)
pp = pprint.PrettyPrinter(indent=4)


class MyTestCase(unittest.TestCase):
    def test_as_supertype_sequential(self):
        input_shape = (8, 8, 3)
        model = Sequential(
            [
                # the bug occurs only when using the "as_supertype_export" function
                # with:
                # - lipschitz coef != 1.0
                # - Frobenius layer ( not the Spectral ones )
                # - Sequential model ( not Model )
                # - tensorflow 2.1 ( not 2.0 )
                Input(input_shape),
                SpectralConv2D(2, (3, 3)),
                ScaledL2NormPooling2D((2, 2)),
                FrobeniusConv2D(2, (3, 3)),
                SpectralConv2DTranspose(5, 3),
                Flatten(),
                Dense(4),
                SpectralDense(4),
                FrobeniusDense(2),
            ],
            k_coef_lip=5.0,
        )
        self._test_model(model, input_shape)

    def test_as_supertype_model(self):
        input_shape = (8, 8, 3)
        inp = Input(input_shape)
        x = SpectralConv2D(2, (3, 3), k_coef_lip=2.0)(inp)
        x = ScaledL2NormPooling2D((2, 2), k_coef_lip=2.0)(x)
        x = FrobeniusConv2D(2, (3, 3), k_coef_lip=2.0)(x)
        x = SpectralConv2DTranspose(5, 3, k_coef_lip=2.0)(x)
        x = Flatten()(x)
        x = Dense(4)(x)
        x = SpectralDense(4, k_coef_lip=2.0)(x)
        out = FrobeniusDense(2, k_coef_lip=2.0)(x)
        model = Model(inputs=inp, outputs=out)
        self._test_model(model, input_shape)

    def _test_model(self, model, input_shape):
        batch_size = 250
        epochs = 1
        steps_per_epoch = 125
        k_lip_data = 2.0
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
        logdir = os.path.join(
            "logs", "lip_layers", "condense_test"
        )  # , datetime.now().strftime("%Y_%m_%d-%H_%M_%S")))
        callback_list = [callbacks.TensorBoard(logdir)]
        # train model
        model.__getattribute__(FIT)(
            linear_generator(batch_size, input_shape, kernel),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=0,
            callbacks=callback_list,
        )
        # the seed is set to compare all models with the same data
        np.random.seed(42)
        # get original results
        loss, mse = model.__getattribute__(EVALUATE)(
            linear_generator(batch_size, input_shape, kernel),
            steps=10,
            verbose=0,
        )
        # generate vanilla
        vanilla_model = model.vanilla_export()
        vanilla_model.compile(
            optimizer=optimizer, loss="mean_squared_error", metrics=[metrics.mse]
        )
        np.random.seed(42)
        # evaluate vanilla
        loss2, mse2 = model.__getattribute__(EVALUATE)(
            linear_generator(batch_size, input_shape, kernel),
            steps=10,
            verbose=0,
        )
        np.random.seed(42)
        # check if original has changed
        vanilla_loss, vanilla_mse = vanilla_model.__getattribute__(EVALUATE)(
            linear_generator(batch_size, input_shape, kernel),
            steps=10,
            verbose=0,
        )
        model.summary()
        vanilla_model.summary()
        self.assertEqual(
            mse,
            vanilla_mse,
            "the exported vanilla model must have same behaviour as original",
        )
        self.assertEqual(mse, mse2, "exporting a model must not change original model")
        # add one epoch to orginal
        model.__getattribute__(FIT)(
            linear_generator(batch_size, input_shape, kernel),
            steps_per_epoch=steps_per_epoch,
            epochs=1,
            verbose=0,
            callbacks=callback_list,
        )
        np.random.seed(42)
        loss3, mse3 = model.__getattribute__(EVALUATE)(
            linear_generator(batch_size, input_shape, kernel),
            steps=10,
            verbose=0,
        )
        # check if vanilla has changed
        np.random.seed(42)
        vanilla_loss2, vanilla_mse2 = vanilla_model.__getattribute__(EVALUATE)(
            linear_generator(batch_size, input_shape, kernel),
            steps=10,
            verbose=0,
        )
        self.assertEqual(
            vanilla_mse,
            vanilla_mse2,
            "exported model must be completely independent from original",
        )
        self.assertNotAlmostEqual(
            mse,
            mse3,
            4,
            "all tests passe but integrity check failed: the test cannot conclude that "
            + "vanilla_export create a distinct model",
        )


if __name__ == "__main__":
    unittest.main()
