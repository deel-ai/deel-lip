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
from tensorflow.keras import backend as K, Input, Model, metrics

if tf.__version__.startswith("2.0"):
    from tensorflow.python.framework.random_seed import set_seed
else:
    set_seed = tf.random.set_seed
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam
from deel.lip.layers import (
    LipschitzLayer,
    SpectralDense,
    SpectralConv2D,
    FrobeniusDense,
    FrobeniusConv2D,
    LorthRegulConv2D,
)
from deel.lip.model import Sequential
from deel.lip.regularizers import OrthDenseRegularizer
from deel.lip.computeLayerSV import (
    computeLayerSV,
)

FIT = "fit_generator" if tf.__version__.startswith("2.0") else "fit"
EVALUATE = "evaluate_generator" if tf.__version__.startswith("2.0") else "evaluate"

pp = pprint.PrettyPrinter(indent=4)

"""
About these tests:
==================

What is tested:
---------------
- layer instantiation
- orthogonality check ( at +-0.001 )

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
    b = layer(a)
    return Model(inputs=a, outputs=b)


class LipschitzLayersSVTest(unittest.TestCase):
    def train_compute_and_verifySV(
        self,
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
        Create a  model, train compute and verify SVs.

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
        """
        flag_test_SVmin = True
        if "dont_test_SVmin" in kwargs.keys():
            flag_test_SVmin = kwargs["dont_test_SVmin"]
        if "k_lip_tolerance_factor" not in kwargs.keys():
            kwargs["k_lip_tolerance_factor"] = 1.02
        # clear session to avoid side effects from previous train
        K.clear_session()
        np.random.seed(42)
        # create the keras model, defin opt, and compile it
        model = generate_k_lip_model(layer_type, layer_params, input_shape, k_lip_model)
        print(model.summary())

        optimizer = Adam(lr=0.001)
        model.compile(
            optimizer=optimizer, loss="mean_squared_error", metrics=[metrics.mse]
        )
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
        callback_list = [hp.KerasCallback(logdir, hparams)]
        if kwargs["callbacks"] is not None:
            callback_list = callback_list + kwargs["callbacks"]
        # train model
        model.__getattribute__(FIT)(
            linear_generator(batch_size, input_shape, kernel),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=1,
            callbacks=callback_list,
        )

        file_writer = tf.summary.create_file_writer(os.path.join(logdir, "metrics"))
        file_writer.set_as_default()
        for ll in model.layers:
            print(ll.name)
            SVmin, SVmax = computeLayerSV(ll)
            # log metrics
            if SVmin is not None:
                tf.summary.text("Layer name", ll.name, step=epochs)
                tf.summary.scalar("SVmin_estim", SVmin, step=epochs)
                tf.summary.scalar("SVmax_estim", SVmax, step=epochs)
                self.assertLess(
                    SVmax,
                    k_lip_model * kwargs["k_lip_tolerance_factor"],
                    msg=" the maximum singular value of the layer "
                    + ll.name
                    + " must be lower than the specified boundary",  # noqa: E501
                )
                self.assertLessEqual(
                    SVmin,
                    SVmax,
                    msg=" the minimum singular value of the layer "
                    + ll.name
                    + " must be lower than the maximum value",  # noqa: E501
                )
                if flag_test_SVmin:
                    self.assertGreater(
                        SVmin,
                        k_lip_model * (2.0 - kwargs["k_lip_tolerance_factor"]),
                        msg=" the minimum singular value of the layer "
                        + ll.name
                        + " must be greater than the specified boundary",  # noqa: E501
                    )
        return

    def _apply_tests_bank(self, tests_bank):
        for test_params in tests_bank:
            pp.pprint(test_params)
            self.train_compute_and_verifySV(**test_params)

    def test_spectral_dense(self):
        self._apply_tests_bank(
            [
                dict(
                    layer_type=SpectralDense,
                    layer_params={
                        "units": 4,
                        "use_bias": False,
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(4,),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
                dict(
                    layer_type=SpectralDense,
                    layer_params={
                        "units": 4,
                    },
                    batch_size=1000,
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
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(4,),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    dont_test_SVmin=False,
                    callbacks=[],
                ),
                dict(
                    layer_type=FrobeniusDense,
                    layer_params={"units": 1},
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(4,),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                    dont_test_SVmin=False,
                    callbacks=[],
                ),
            ]
        )

    def test_orthRegul_dense(self):
        """
        Tests for a standard Dense layer, for result comparison.
        """
        self._apply_tests_bank(
            [
                dict(
                    layer_type=Dense,
                    layer_params={
                        "units": 6,
                        "kernel_regularizer": OrthDenseRegularizer(1000.0),
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=10,
                    input_shape=(4,),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
            ]
        )

    # dict(
    #     layer_type=Dense,
    #     layer_params={"units": 4, "kernel_regularizer": OrthDenseRegularizer(1000.0)},
    #     batch_size=1000,
    #     steps_per_epoch=125,
    #     epochs=5,
    #     input_shape=(4,),
    #     k_lip_data=1.0,
    #     k_lip_model=5.0,
    #     callbacks=[],
    # ),  # No k factor in dense layer

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
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    k_lip_tolerance_factor=1.1,
                    dont_test_SVmin=False,
                    callbacks=[],
                ),
                dict(
                    layer_type=SpectralConv2D,
                    layer_params={"filters": 2, "kernel_size": (3, 3)},
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                    k_lip_tolerance_factor=1.1,
                    dont_test_SVmin=False,
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
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    dont_test_SVmin=False,
                    callbacks=[],
                ),
                dict(
                    layer_type=FrobeniusConv2D,
                    layer_params={"filters": 2, "kernel_size": (3, 3)},
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=5,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                    dont_test_SVmin=False,
                    callbacks=[],
                ),
            ]
        )

    def test_lorthregulconv2d(self):
        # tests only checks that lip cons is enforced
        self._apply_tests_bank(
            [
                dict(
                    layer_type=LorthRegulConv2D,
                    layer_params={
                        "filters": 2,
                        "kernel_size": (3, 3),
                        "lambdaLorth": 1000.0,
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=10,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    k_lip_tolerance_factor=1.1,
                    callbacks=[],
                ),
                dict(
                    layer_type=LorthRegulConv2D,
                    layer_params={
                        "filters": 2,
                        "kernel_size": (3, 3),
                        "lambdaLorth": 1000.0,
                    },
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=10,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=5.0,
                    k_lip_tolerance_factor=1.1,
                    callbacks=[],
                ),
            ]
        )

    """
    def test_scaledaveragepooling2d(self):
        # tests only checks that lip cons is enforced
        self._apply_tests_bank(
            [
                dict(
                    layer_type=ScaledAveragePooling2D,
                    layer_params={"pool_size": (3, 3)},
                    batch_size=1000,
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
                    batch_size=1000,
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
                    batch_size=1000,
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
                    batch_size=1000,
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
                    batch_size=1000,
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
                    batch_size=1000,
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
                    batch_size=1000,
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
                    batch_size=1000,
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
                    batch_size=1000,
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
                    batch_size=1000,
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
                    batch_size=1000,
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
                    batch_size=1000,
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
                    batch_size=1000,
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
                    batch_size=1000,
                    steps_per_epoch=125,
                    epochs=10,
                    input_shape=(5, 5, 1),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[
                        # CondenseCallback(on_batch=False, on_epoch=True),
                        MonitorCallback(
                            monitored_layers=["conv1", "dense1"],
                            logdir=os.path.join("logs", "lip_layers", "Sequential"),
                            what="max",
                            on_epoch=False,
                            on_batch=True,
                        )
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
                    batch_size=1000,
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
                    batch_size=1000,
                    steps_per_epoch=1,
                    epochs=5,
                    input_shape=(6, 6, 18),
                    k_lip_data=1.0,
                    k_lip_model=1.0,
                    callbacks=[],
                ),
            ]
        )
        """


"""import tensorflow as tf
from  deel.lip.computeLayerSV import  generate_graph_layers

model = tf.keras.applications.ResNet50()
model.summary()
list_SV = computeModelUpperLip(model)
print(list_SV)
generate_graph_layers(model,layerName=model.layers[0].name)"""

if __name__ == "__main__":

    unittest.main()
