# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import os
import pprint
import tempfile
import unittest

import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import backend as K, Input, Model, metrics, callbacks

from deel.lip.constraints import (
    AutoWeightClipConstraint,
    SpectralConstraint,
    FrobeniusConstraint,
)

if tf.__version__.startswith("2.0"):
    from tensorflow.python.framework.random_seed import set_seed
else:
    set_seed = tf.random.set_seed
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from deel.lip.callbacks import CondenseCallback, MonitorCallback
from deel.lip.layers import (
    LipschitzLayer,
    SpectralDense,
    SpectralConv2D,
    SpectralConv2DTranspose,
    FrobeniusDense,
    FrobeniusConv2D,
    OrthoConv2D,
    ScaledAveragePooling2D,
    ScaledGlobalAveragePooling2D,
    ScaledL2NormPooling2D,
    InvertibleDownSampling,
    InvertibleUpSampling,
    ScaledGlobalL2NormPooling2D,
)
from deel.lip.model import Sequential
from deel.lip.utils import evaluate_lip_const

FIT = "fit_generator" if tf.__version__.startswith("2.0") else "fit"
EVALUATE = "evaluate_generator" if tf.__version__.startswith("2.0") else "evaluate"

pp = pprint.PrettyPrinter(indent=4)

"""
About these tests:
==================

What is tested:
---------------
- layer instantiation
- training
- prediction
- storing on disk and reloading
- k lip_constraint is respected ( at +-0.001 )

What is not tested:
-------------------
- layer performance ( time / accuracy )
- layer structure ( don't check that SpectralConv2D is actually a convolution )

However, all run generate log that can be manually checked with tensorboard
"""


def linear_generator(batch_size, input_shape: tuple, kernel):
    """
    Generate data according to a linear kernel
    Args:
        batch_size: size of each batch
        input_shape: shape of the desired input
        kernel: kernel used to generate data, must match the last dimensions of
            `input_shape`

    Returns:
        a generator for the data

    """
    input_shape = tuple(input_shape)
    while True:
        # pick random sample in [0, 1] with the input shape
        batch_x = np.array(
            np.random.uniform(-10, 10, (batch_size,) + input_shape), dtype=np.float16
        )
        # apply the k lip linear transformation
        batch_y = np.tensordot(
            batch_x,
            kernel,
            axes=(
                [i for i in range(1, len(input_shape) + 1)],
                [i for i in range(0, len(input_shape))],
            ),
        )
        yield batch_x, batch_y


def build_kernel(input_shape: tuple, output_shape: tuple, k=1.0):
    """
    build a kernel with defined lipschitz factor

    Args:
        input_shape: input shape of the linear function
        output_shape: output shape of the linear function
        k: lipshitz factor of the function

    Returns:
        the kernel for use in the linear_generator

    """
    input_shape = tuple(input_shape)
    output_shape = tuple(output_shape)
    kernel = np.array(
        np.random.random_sample(input_shape + output_shape), dtype=np.float16
    )
    kernel = (
        kernel * k / np.linalg.norm(kernel)
    )  # assuming lipschitz constraint is independent with respect to the chosen metric

    return kernel


def generate_k_lip_model(layer_type: type, layer_params: dict, input_shape, k):
    """
    build a model with a single layer of given type, with defined lipshitz factor.

    Args:
        layer_type: the type of layer to use
        layer_params: parameter passed to constructor of layer_type
        input_shape: the shape of the input
        k: lipshitz factor of the function

    Returns:
        a keras Model with a single layer.

    """
    if issubclass(layer_type, Sequential):
        model = layer_type(**layer_params)
        model.set_klip_factor(k)
        return model
    a = Input(shape=input_shape)
    if issubclass(layer_type, LipschitzLayer):
        layer_params["k_coef_lip"] = k
    layer = layer_type(**layer_params)
    assert isinstance(layer, Layer)
    # print(input_shape)
    # print(layer.compute_output_shape((32, ) + input_shape))
    b = layer(a)
    return Model(inputs=a, outputs=b)


def train_k_lip_model(
    layer_type: type,
    layer_params: dict,
    batch_size: int,
    steps_per_epoch: int,
    epochs: int,
    input_shape: tuple,
    k_lip_model: float,
    k_lip_data: float,
    **kwargs
):
    """
    Create a generator, create a model, train it and return the results.

    Args:
        layer_type:
        layer_params:
        batch_size:
        steps_per_epoch:
        epochs:
        input_shape:
        k_lip_model:
        k_lip_data:
        **kwargs:

    Returns:
        the generator

    """
    # clear session to avoid side effects from previous train
    K.clear_session()
    np.random.seed(42)
    # create the keras model, defin opt, and compile it
    model = generate_k_lip_model(layer_type, layer_params, input_shape, k_lip_model)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=[metrics.mse])
    # model.summary()
    # create the synthetic data generator
    output_shape = model.compute_output_shape((batch_size,) + input_shape)[1:]
    kernel = build_kernel(input_shape, output_shape, k_lip_data)
    # define logging features
    logdir = os.path.join("logs", "lip_layers", "%s" % layer_type.__name__)
    hparams = dict(
        layer_type=layer_type.__name__,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        k_lip_data=k_lip_data,
        k_lip_model=k_lip_model,
    )
    callback_list = [callbacks.TensorBoard(logdir), hp.KerasCallback(logdir, hparams)]
    if kwargs["callbacks"] is not None:
        callback_list = callback_list + kwargs["callbacks"]
    # train model
    model.__getattribute__(FIT)(
        linear_generator(batch_size, input_shape, kernel),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=0,
        callbacks=callback_list,
    )
    # the seed is set to compare all models with the same data
    x, y = linear_generator(batch_size, input_shape, kernel).send(None)
    np.random.seed(42)
    set_seed(42)
    loss, mse = model.__getattribute__(EVALUATE)(
        linear_generator(batch_size, input_shape, kernel),
        steps=10,
    )
    empirical_lip_const = evaluate_lip_const(model=model, x=x, seed=42)
    # save the model
    model_checkpoint_path = os.path.join(logdir, "model.keras")
    model.save(model_checkpoint_path, overwrite=True)
    del model
    K.clear_session()
    model = load_model(model_checkpoint_path)
    np.random.seed(42)
    set_seed(42)
    from_disk_loss, from_disk_mse = model.__getattribute__(EVALUATE)(
        linear_generator(batch_size, input_shape, kernel),
        steps=10,
    )
    from_empirical_lip_const = evaluate_lip_const(model=model, x=x, seed=42)
    # log metrics
    file_writer = tf.summary.create_file_writer(os.path.join(logdir, "metrics"))
    file_writer.set_as_default()
    tf.summary.scalar("lip_coef_estim", empirical_lip_const, step=epochs)
    tf.summary.scalar("evaluation_mse", mse, step=epochs)
    tf.summary.scalar("disk_load_evaluation_mse", from_disk_mse, step=epochs)
    tf.summary.scalar("disk_load_lip_coef_estim", from_empirical_lip_const, step=epochs)
    return (
        mse,
        empirical_lip_const.numpy(),
        from_disk_mse,
        from_empirical_lip_const.numpy(),
    )


class LipschitzLayersTest(unittest.TestCase):
    def _check_mse_results(self, mse, from_disk_mse, test_params):
        self.assertAlmostEqual(
            mse,
            from_disk_mse,
            5,
            "serialization must not change the performance of a layer",
        )

    def _check_emp_lip_const(self, emp_lip_const, from_disk_emp_lip_const, test_params):
        self.assertAlmostEqual(
            emp_lip_const,
            from_disk_emp_lip_const,
            5,
            "serialization must not change the Lipschitz constant of a layer",
        )
        self.assertLess(
            emp_lip_const,
            test_params["k_lip_model"] * 1.02,
            msg=" the lip const of the network must be lower"
            + " than the specified boundary",
        )

    def _apply_tests_bank(self, tests_bank):
        for test_params in tests_bank:
            pp.pprint(test_params)
            (
                mse,
                emp_lip_const,
                from_disk_mse,
                from_disk_emp_lip_const,
            ) = train_k_lip_model(**test_params)
            print("test mse: %f" % mse)
            print(
                "empirical lip const: %f ( expected %s )"
                % (
                    emp_lip_const,
                    min(test_params["k_lip_model"], test_params["k_lip_data"]),
                )
            )
            self._check_mse_results(mse, from_disk_mse, test_params)
            self._check_emp_lip_const(
                emp_lip_const, from_disk_emp_lip_const, test_params
            )

    def test_constraints_clipping(self):
        """
        Tests for a standard Dense layer, for result comparison.
        """
        self._apply_tests_bank(
            [
                dict(
                    layer_type=Dense,
                    layer_params={
                        "units": 4,
                        "kernel_constraint": AutoWeightClipConstraint(1.0),
                    },
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(4,),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=Dense,
                    layer_params={
                        "units": 4,
                        "kernel_constraint": AutoWeightClipConstraint(1.0),
                    },
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(4,),
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=Dense,
                    layer_params={
                        "units": 4,
                        "kernel_constraint": AutoWeightClipConstraint(5.0),
                    },
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(4,),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                    callbacks=[],
                ),
            ]
        )

    def test_constraints_orthogonal(self):
        """
        Tests for a standard Dense layer, for result comparison.
        """
        self._apply_tests_bank(
            [
                dict(
                    layer_type=Dense,
                    layer_params={
                        "units": 4,
                        "kernel_constraint": SpectralConstraint(1.0),
                    },
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(4,),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=Dense,
                    layer_params={
                        "units": 4,
                        "kernel_constraint": SpectralConstraint(1.0),
                    },
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(4,),
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=Dense,
                    layer_params={
                        "units": 4,
                        "kernel_constraint": SpectralConstraint(5.0),
                    },
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(4,),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                    callbacks=[],
                ),
            ]
        )

    def test_constraints_frobenius(self):
        """
        Tests for a standard Dense layer, for result comparison.
        """
        self._apply_tests_bank(
            [
                dict(
                    layer_type=Dense,
                    layer_params={
                        "units": 4,
                        "kernel_constraint": FrobeniusConstraint(),
                    },
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(4,),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=Dense,
                    layer_params={
                        "units": 4,
                        "kernel_constraint": FrobeniusConstraint(),
                    },
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(4,),
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
            ]
        )

    def test_spectral_dense(self):
        self._apply_tests_bank(
            [
                dict(
                    layer_type=SpectralDense,
                    layer_params={"units": 3, "use_bias": False},
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(4,),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=SpectralDense,
                    layer_params={"units": 4},
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(4,),
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=SpectralDense,
                    layer_params={"units": 4},
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(4,),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                    callbacks=[],
                ),
            ]
        )

    def test_frobenius_dense(self):
        self._apply_tests_bank(
            [
                dict(
                    layer_type=FrobeniusDense,
                    layer_params={"units": 1},
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(4,),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=FrobeniusDense,
                    layer_params={"units": 1},
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(4,),
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=FrobeniusDense,
                    layer_params={"units": 1},
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(4,),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                    callbacks=[],
                ),
            ]
        )

    def test_spectralconv2d(self):
        self._apply_tests_bank(
            [
                dict(
                    layer_type=SpectralConv2D,
                    layer_params={
                        "filters": 2,
                        "kernel_size": (3, 3),
                        "use_bias": False,
                    },
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=SpectralConv2D,
                    layer_params={"filters": 2, "kernel_size": (3, 3)},
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(5, 5, 1),
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=SpectralConv2D,
                    layer_params={"filters": 2, "kernel_size": (3, 3)},
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                    callbacks=[],
                ),
            ]
        )

    def test_SpectralConv2DTranspose(self):
        self._apply_tests_bank(
            [
                dict(
                    layer_type=SpectralConv2DTranspose,
                    layer_params={
                        "filters": 2,
                        "kernel_size": (3, 3),
                        "use_bias": False,
                    },
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=SpectralConv2DTranspose,
                    layer_params={"filters": 2, "kernel_size": (3, 3)},
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(5, 5, 1),
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=SpectralConv2DTranspose,
                    layer_params={"filters": 2, "kernel_size": (3, 3)},
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                    callbacks=[],
                ),
            ]
        )

    def test_frobeniusconv2d(self):
        # tests only checks that lip cons is enforced
        self._apply_tests_bank(
            [
                dict(
                    layer_type=FrobeniusConv2D,
                    layer_params={"filters": 2, "kernel_size": (3, 3)},
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=FrobeniusConv2D,
                    layer_params={"filters": 2, "kernel_size": (3, 3)},
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(5, 5, 1),
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=FrobeniusConv2D,
                    layer_params={"filters": 2, "kernel_size": (3, 3)},
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                    callbacks=[],
                ),
            ]
        )

    def test_Orthoconv2d(self):
        # tests only checks that lip cons is enforced
        self._apply_tests_bank(
            [
                dict(
                    layer_type=OrthoConv2D,
                    layer_params={
                        "filters": 2,
                        "kernel_size": (3, 3),
                        "regul_lorth": 1000.0,
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=10,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    k_lip_tolerance_factor=1.2,
                    callbacks=[],
                ),
                dict(
                    layer_type=OrthoConv2D,
                    layer_params={
                        "filters": 2,
                        "kernel_size": (3, 3),
                        "regul_lorth": 1000.0,
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=10,
                    input_shape=(5, 5, 1),
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                    k_lip_tolerance_factor=1.2,
                    callbacks=[],
                ),
                dict(
                    layer_type=OrthoConv2D,
                    layer_params={
                        "filters": 2,
                        "kernel_size": (3, 3),
                        "regul_lorth": 1000.0,
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=10,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                    k_lip_tolerance_factor=1.2,
                    callbacks=[],
                ),
                dict(  # No Regul only spectral norm
                    layer_type=OrthoConv2D,
                    layer_params={
                        "filters": 2,
                        "kernel_size": (3, 3),
                        "regul_lorth": 0.0,
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=10,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    k_lip_tolerance_factor=1.2,
                    callbacks=[],
                ),
            ]
        )

    def test_scaledaveragepooling2d(self):
        # tests only checks that lip cons is enforced
        self._apply_tests_bank(
            [
                dict(
                    layer_type=ScaledAveragePooling2D,
                    layer_params={"pool_size": (3, 3)},
                    batch_size=250,
                    steps_per_epoch=1,
                    epochs=1,
                    input_shape=(6, 6, 1),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=ScaledAveragePooling2D,
                    layer_params={"pool_size": (3, 3)},
                    batch_size=250,
                    steps_per_epoch=1,
                    epochs=1,
                    input_shape=(6, 6, 1),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=ScaledAveragePooling2D,
                    layer_params={"pool_size": (3, 3)},
                    batch_size=250,
                    steps_per_epoch=1,
                    epochs=1,
                    input_shape=(6, 6, 1),
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
            ]
        )

    def test_scaledglobalaveragepooling2d(self):
        self._apply_tests_bank(
            [
                # tests only checks that lip cons is enforced
                dict(
                    layer_type=Sequential,
                    layer_params={
                        "layers": [
                            Input((5, 5, 1)),
                            ScaledGlobalAveragePooling2D(data_format="channels_last"),
                        ]
                    },
                    batch_size=250,
                    steps_per_epoch=1,
                    epochs=1,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=Sequential,
                    layer_params={
                        "layers": [
                            Input((5, 5, 1)),
                            ScaledGlobalAveragePooling2D(data_format="channels_last"),
                        ]
                    },
                    batch_size=250,
                    steps_per_epoch=1,
                    epochs=1,
                    input_shape=(5, 5, 1),
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=Sequential,
                    layer_params={
                        "layers": [
                            Input((5, 5, 1)),
                            ScaledGlobalAveragePooling2D(data_format="channels_last"),
                        ]
                    },
                    batch_size=250,
                    steps_per_epoch=1,
                    epochs=1,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                    callbacks=[],
                ),
            ]
        )

    def test_scaledl2normpooling2d(self):
        self._apply_tests_bank(
            [
                # tests only checks that lip cons is enforced
                dict(
                    layer_type=Sequential,
                    layer_params={
                        "layers": [
                            Input((5, 5, 1)),
                            ScaledL2NormPooling2D((2, 3), data_format="channels_last"),
                        ]
                    },
                    batch_size=250,
                    steps_per_epoch=1,
                    epochs=1,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=Sequential,
                    layer_params={
                        "layers": [
                            Input((5, 5, 1)),
                            ScaledL2NormPooling2D((2, 3), data_format="channels_last"),
                        ]
                    },
                    batch_size=250,
                    steps_per_epoch=1,
                    epochs=1,
                    input_shape=(5, 5, 1),
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=Sequential,
                    layer_params={
                        "layers": [
                            Input((5, 5, 1)),
                            ScaledL2NormPooling2D((2, 3), data_format="channels_last"),
                        ]
                    },
                    batch_size=250,
                    steps_per_epoch=1,
                    epochs=1,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                    callbacks=[],
                ),
            ]
        )

    def test_scaledgloball2normpooling2d(self):
        self._apply_tests_bank(
            [
                # tests only checks that lip cons is enforced
                dict(
                    layer_type=Sequential,
                    layer_params={
                        "layers": [
                            Input((5, 5, 1)),
                            ScaledGlobalL2NormPooling2D(data_format="channels_last"),
                        ]
                    },
                    batch_size=250,
                    steps_per_epoch=1,
                    epochs=1,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=Sequential,
                    layer_params={
                        "layers": [
                            Input((5, 5, 1)),
                            ScaledGlobalL2NormPooling2D(data_format="channels_last"),
                        ]
                    },
                    batch_size=250,
                    steps_per_epoch=1,
                    epochs=1,
                    input_shape=(5, 5, 1),
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=Sequential,
                    layer_params={
                        "layers": [
                            Input((5, 5, 1)),
                            ScaledGlobalL2NormPooling2D(data_format="channels_last"),
                        ]
                    },
                    batch_size=250,
                    steps_per_epoch=1,
                    epochs=1,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                    callbacks=[],
                ),
            ]
        )

    def test_multilayer(self):
        self._apply_tests_bank(
            [
                dict(
                    layer_type=Sequential,
                    layer_params={
                        "layers": [
                            Input((5, 5, 1)),
                            SpectralConv2D(2, (3, 3), use_bias=False),
                            SpectralDense(4),
                        ]
                    },
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=Sequential,
                    layer_params={
                        "layers": [
                            Input((5, 5, 1)),
                            SpectralConv2D(2, (3, 3)),
                            SpectralDense(4),
                        ]
                    },
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(5, 5, 1),
                    k_lip_data=5.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=Sequential,
                    layer_params={
                        "layers": [
                            Input((5, 5, 1)),
                            SpectralConv2D(2, (3, 3)),
                            SpectralDense(4),
                        ]
                    },
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                    callbacks=[],
                ),
            ]
        )

    def test_callbacks(self):
        self._apply_tests_bank(
            [
                dict(
                    layer_type=Sequential,
                    layer_params={
                        "layers": [
                            Input((5, 5, 1)),
                            SpectralConv2D(2, (3, 3)),
                            ScaledAveragePooling2D((2, 2)),
                            SpectralDense(4),
                        ]
                    },
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[CondenseCallback(on_batch=True, on_epoch=False)],
                ),
                dict(
                    layer_type=Sequential,
                    layer_params={
                        "layers": [
                            Input((5, 5, 1)),
                            SpectralConv2D(2, (3, 3), name="conv1"),
                            ScaledAveragePooling2D((2, 2)),
                            SpectralDense(4, name="dense1"),
                        ]
                    },
                    batch_size=250,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[
                        # CondenseCallback(on_batch=False, on_epoch=True),
                        MonitorCallback(
                            monitored_layers=["conv1", "dense1"],
                            logdir=os.path.join("logs", "lip_layers", "Sequential"),
                            target="kernel",
                            what="all",
                            on_epoch=False,
                            on_batch=True,
                        ),
                        MonitorCallback(
                            monitored_layers=["conv1", "dense1"],
                            logdir=os.path.join("logs", "lip_layers", "Sequential"),
                            target="wbar",
                            what="all",
                            on_epoch=False,
                            on_batch=True,
                        ),
                    ],
                ),
            ]
        )

    def test_invertibledownsampling(self):
        # tests only checks that lip cons is enforced
        self._apply_tests_bank(
            [
                dict(
                    layer_type=InvertibleDownSampling,
                    layer_params={"pool_size": (2, 3)},
                    batch_size=250,
                    steps_per_epoch=1,
                    epochs=5,
                    input_shape=(6, 6, 3),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
            ]
        )

    def test_invertibleupsampling(self):
        # tests only checks that lip cons is enforced
        self._apply_tests_bank(
            [
                dict(
                    layer_type=InvertibleUpSampling,
                    layer_params={"pool_size": (2, 3)},
                    batch_size=250,
                    steps_per_epoch=1,
                    epochs=5,
                    input_shape=(6, 6, 18),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
            ]
        )


class TestSpectralConv2DTranspose(unittest.TestCase):
    def test_instantiation(self):
        # Supported cases
        cases = (
            dict(filters=5, kernel_size=3),
            dict(filters=12, kernel_size=5, strides=2, use_bias=False),
            dict(filters=3, kernel_size=3, padding="same", dilation_rate=1),
            dict(filters=4, kernel_size=1, output_padding=None, activation="relu"),
            dict(filters=16, kernel_size=3, data_format="channels_first"),
        )

        for i, kwargs in enumerate(cases):
            with self.subTest(i=i):
                SpectralConv2DTranspose(**kwargs)

        # Unsupported cases
        cases = (
            {"msg": "Wrong padding", "kwarg": {"padding": "valid"}},
            {"msg": "Wrong dilation rate", "kwarg": {"dilation_rate": 2}},
            {"msg": "Wrong data format", "kwarg": {"output_padding": 5}},
        )

        for case in cases:
            with self.subTest(case["msg"]):
                with self.assertRaises(ValueError):
                    SpectralConv2DTranspose(10, 3, **case["kwarg"])

    def test_vanilla_export(self):
        kwargs = dict(
            filters=16,
            kernel_size=5,
            strides=2,
            activation="relu",
            data_format="channels_first",
            input_shape=(28, 28, 3),
        )

        lay = SpectralConv2DTranspose(**kwargs)
        model = Sequential([lay])

        x = tf.random.normal((5,) + (kwargs["input_shape"]))
        y1 = model(x)

        # Test vanilla export inference comparison
        vanilla_model = model.vanilla_export()
        y2 = vanilla_model(x)
        np.testing.assert_allclose(y1, y2, atol=1e-6)

        # Test saving/loading model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.keras")
            model.save(model_path)
            tf.keras.models.load_model(model_path)


if __name__ == "__main__":
    unittest.main()
