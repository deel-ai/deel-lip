# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import os
import pprint
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

from deel.lip.utils import padding_circular

if tf.__version__.startswith("2.0"):
    from tensorflow.python.framework.random_seed import set_seed
else:
    set_seed = tf.random.set_seed
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from deel.lip.callbacks import CondenseCallback, MonitorCallback
from deel.lip.layers import (
    LipschitzLayer,
    SpectralDense,
    PadConv2D,
    SpectralConv2D,
    FrobeniusDense,
    FrobeniusConv2D,
    ScaledAveragePooling2D,
    ScaledGlobalAveragePooling2D,
    ScaledL2NormPooling2D,
    InvertibleDownSampling,
    InvertibleUpSampling,
    ScaledGlobalL2NormPooling2D,
)
from deel.lip.model import Sequential, vanillaModel
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
    optimizer = Adam(lr=0.001)
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
    model_checkpoint_path = os.path.join(logdir, "model.h5")
    model.save(model_checkpoint_path, overwrite=True, save_format="h5")
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

class TestPadConv2D(unittest.TestCase):
    def test_PadConv2D(self):
        layer_params={
            "filters": 2,
            "kernel_size": (3, 3),
            "use_bias": False,
            "padding": "valid"
        }
        padding_tested = "circular"
        batch_size=250
        kernel_input_shapes=[
            [(3,3),2,(5, 5, 1)],
            [(3,3),2,(5,5,5)],
            [(7,7),2,(7,7,2)],
            [3,2,(5,5,128)]
            ]
        for k_shape, filters, input_shape in kernel_input_shapes:
            layer_params["kernel_size"] = k_shape
            layer_params["filters"] = filters
            for pad in ["circular","constant", "symmetric", "reflect","same", "valid"]:
                self._test_predict(layer_params,pad,input_shape,batch_size)
            for pad in ["circular","constant", "symmetric", "reflect"]:
                self._test_padding(pad,input_shape,batch_size,layer_params["kernel_size"])
            for pad in ["circular","constant", "symmetric", "reflect","same", "valid"]:
                self._test_vanilla(layer_params,pad,input_shape,batch_size)
        
    def pad_input(self,x,padding,kernel_size):
        if isinstance(kernel_size, (int, float)):
            kernel_size = [kernel_size,kernel_size]
        if padding.lower() in ["same", "valid"]:
            pad = lambda x: x
        if padding.upper() in ["CONSTANT", "REFLECT", "SYMMETRIC"]:
            p_vert, p_hor = kernel_size[0] // 2, kernel_size[1] // 2
            paddings = [[0, 0], [p_vert, p_vert], [p_hor, p_hor], [0, 0]]
            pad = lambda t: tf.pad(t, paddings, padding)
        if padding.lower() in ["circular"]:
            p_vert, p_hor = kernel_size[0] // 2, kernel_size[1] // 2
            pad = lambda t: padding_circular(t, (p_vert, p_hor))
        x = pad(x)
        return x

    def compare(self,x,x_ref,index_x=[],index_x_ref=[]):
        print("\tcheck "+index_x[-1])
        #print(index_x_ref)
        x_cropped =  x[:,index_x[0]:index_x[1],index_x[3]:index_x[4],:][:,::index_x[2],::index_x[5],:] 
        if index_x_ref[0]==None: #compare with 0
            np.testing.assert_allclose(
                x_cropped, np.zeros(x_cropped.shape), 1e-2, 0
            )
        else:
            print(index_x)
            #print(x_cropped[0,:,:,0])
            print(index_x_ref)
            #print(x_ref[0,index_x_ref[0]:index_x_ref[1],index_x_ref[3]:index_x_ref[4],0])
            #print(x_ref[:,index_x_ref[0]:index_x_ref[1],index_x_ref[3]:index_x_ref[4],:][0,::index_x_ref[2],::index_x_ref[5],0])
            np.testing.assert_allclose(
                x_cropped, x_ref[:,index_x_ref[0]:index_x_ref[1],index_x_ref[3]:index_x_ref[4],:][:,::index_x_ref[2],::index_x_ref[5],:], 1e-2, 0
            )

    def _test_padding(self, padding_tested, input_shape, batch_size, kernel_size):
        print(padding_tested)
        print(kernel_size)

        kernel_size_list= kernel_size
        if isinstance(kernel_size, (int, float)):
            kernel_size_list = [kernel_size,kernel_size]

        x = np.random.normal(size=(batch_size,)+input_shape).astype('float32')
        x_pad = self.pad_input(x,padding_tested,kernel_size)
        p_vert, p_hor = kernel_size_list[0] // 2, kernel_size_list[1] // 2
        
        center_x_pad = [p_vert,-p_vert,1,p_hor,-p_hor,1,"center"]
        upper_x_pad = [0,p_vert,1,p_hor,-p_hor,1,"upper"]
        lower_x_pad = [-p_vert,x_pad.shape[1],1,p_hor,-p_hor,1,"lower"]
        left_x_pad = [p_vert,-p_vert,1,0,p_hor,1,"left"]
        right_x_pad = [p_vert,-p_vert,1,-p_hor,x_pad.shape[2],1,"right"]
        all_x = [0,x.shape[1],1,0,x.shape[2],1]
        upper_x = [0,p_vert,1,0,x.shape[2],1]
        upper_x_rev = [0,p_vert,-1,0,x.shape[2],1]
        upper_x_refl = [1,p_vert+1,-1,0,x.shape[2],1]
        lower_x = [-p_vert,x.shape[1],1,0,x.shape[2],1]
        lower_x_rev = [-p_vert,x.shape[1],-1,0,x.shape[2],1]
        lower_x_refl = [-p_vert-1,x.shape[1]-1,-1,0,x.shape[2],1]
        left_x = [0,x.shape[1],1,0,p_hor,1]
        left_x_rev = [0,x.shape[1],1,0,p_hor,-1]
        left_x_refl = [0,x.shape[1],1,1,p_hor+1,-1]
        right_x = [0,x.shape[1],1,-p_hor,x.shape[2],1]
        right_x_rev = [0,x.shape[1],1,-p_hor,x.shape[2],-1]
        right_x_refl = [0,x.shape[1],1,-p_hor-1,x.shape[2]-1,-1]
        zero_pad = [None,None,None,None]
        pad_tests = [
            {"circular":[center_x_pad,all_x],"constant":[center_x_pad,all_x], "symmetric":[center_x_pad,all_x], "reflect":[center_x_pad,all_x]},
            {"circular":[upper_x_pad,lower_x],"constant":[upper_x_pad,zero_pad], "symmetric":[upper_x_pad,upper_x_rev], "reflect":[upper_x_pad,upper_x_refl]},
            {"circular":[lower_x_pad,upper_x],"constant":[lower_x_pad,zero_pad], "symmetric":[lower_x_pad,lower_x_rev], "reflect":[lower_x_pad,lower_x_refl]},
            {"circular":[left_x_pad,right_x],"constant":[left_x_pad,zero_pad], "symmetric":[left_x_pad,left_x_rev], "reflect":[left_x_pad,left_x_refl]},
            {"circular":[right_x_pad,left_x],"constant":[right_x_pad,zero_pad], "symmetric":[right_x_pad,right_x_rev], "reflect":[right_x_pad,right_x_refl]},
        ]

        for test_pad in pad_tests:
            self.compare(x_pad,x,
                index_x=test_pad[padding_tested][0],
                index_x_ref=test_pad[padding_tested][1])
            
        return

    def _test_predict(self, layer_params, padding_tested, input_shape, batch_size):
        print(layer_params)
        x = np.random.normal(size=(batch_size,)+input_shape).astype('float32')
        print(x.shape)
        x_pad = self.pad_input(x,padding_tested,layer_params["kernel_size"])
        print(x_pad.shape)
        layer_params_ref = layer_params.copy()
        if padding_tested.lower() == "same":
           layer_params_ref["padding"]= padding_tested
        model_ref = generate_k_lip_model(layer_type=Conv2D, layer_params=layer_params_ref, input_shape=x_pad.shape[1:], k=1.0)
        optimizer_ref = Adam(lr=0.001)
        model_ref.compile(optimizer=optimizer_ref, loss="mean_squared_error", metrics=[metrics.mse])
        y_ref = model_ref.predict(x_pad)
        layer_params_pad = layer_params.copy()
        layer_params_pad["padding"] = padding_tested
        print(layer_params_pad)
        model = generate_k_lip_model(layer_type=PadConv2D, layer_params=layer_params_pad, input_shape=input_shape, k=1.0)
        optimizer = Adam(lr=0.001)
        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=[metrics.mse])
        model.summary()

        model.layers[1].kernel.assign(model_ref.layers[1].kernel)
        if model.layers[1].use_bias:
            model.layers[1].bias.assign(model_ref.layers[1].bias)
        
        y = model.predict(x)
        np.testing.assert_allclose(
            y_ref, y, 1e-2, 0
        )

    def _test_vanilla(self, layer_params, padding_tested, input_shape, batch_size):
        print(padding_tested)
        #print(layer_params)
        x = np.random.normal(size=(batch_size,)+input_shape).astype('float32')
        print(x.shape)
        layer_params_pad = layer_params.copy()
        layer_params_pad["padding"] = padding_tested
        print(layer_params_pad)
        model = generate_k_lip_model(layer_type=PadConv2D, layer_params=layer_params_pad, input_shape=input_shape, k=1.0)
        optimizer = Adam(lr=0.001)
        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=[metrics.mse])
        
        #model.layers[1].kernel.assign(model_ref.layers[1].kernel)
        #if model.layers[1].use_bias:
        #    model.layers[1].bias.assign(model_ref.layers[1].bias)
        
        y = model.predict(x)
        model_v = vanillaModel(model)
        optimizer_v = Adam(lr=0.001)
        model_v.compile(optimizer=optimizer_v, loss="mean_squared_error", metrics=[metrics.mse])
        
        y_v = model_v.predict(x)

        np.testing.assert_allclose(
            y_v, y, 1e-2, 0
        )

if __name__ == "__main__":
    unittest.main()
