# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""Tests for singular value computation (in compute_layer_sv.py)"""
import os
import pprint
import pytest

import numpy as np

from .utils_framework import compute_layer_sv
from .utils_framework import (
    tLinear,
    FrobeniusConv2d,
    FrobeniusLinear,
    SpectralConv2d,
    SpectralLinear,
)
from .utils_framework import OrthLinearRegularizer

from . import utils_framework as uft


pp = pprint.PrettyPrinter(indent=4)


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


def train_compute_and_verifySV(
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
    if "test_SVmin" in kwargs.keys():
        flag_test_SVmin = kwargs["test_SVmin"]
    if "k_lip_tolerance_factor" not in kwargs.keys():
        kwargs["k_lip_tolerance_factor"] = 1.02
    # clear session to avoid side effects from previous train
    uft.init_session()  # K.clear_session()
    np.random.seed(42)
    input_shape = uft.to_framework_channel(input_shape)

    # tf.random.uft.set_seed(1234)
    # create the keras model, defin opt, and compile it
    model = uft.generate_k_lip_model(layer_type, layer_params, input_shape, k_lip_model)

    optimizer = uft.get_instance_framework(
        uft.Adam, inst_params={"lr": 0.001, "model": model}
    )
    # optimizer = uft.Adam(lr=0.001)
    loss_fn, optimizer, metrics = uft.compile_model(
        model,
        optimizer=optimizer,
        loss=uft.MeanSquaredError(),
        metrics=[uft.metric_mse()],
    )
    # model.compile(
    #     optimizer=optimizer, loss="mean_squared_error", metrics=[metrics.mse]
    # )
    # create the synthetic data generator
    output_shape = uft.compute_output_shape(input_shape, model)
    # output_shape = model.uft.compute_output_shape((batch_size,) + input_shape)[1:]
    kernel = build_kernel(input_shape, output_shape, k_lip_data)
    # define logging features
    logdir = os.path.join("logs", uft.LIP_LAYERS, "%s" % layer_type.__name__)
    os.makedirs(logdir, exist_ok=True)

    callback_list = []  # [hp.KerasCallback(logdir, hparams)]
    if "callbacks" in kwargs and (kwargs["callbacks"] is not None):
        callback_list = callback_list + kwargs["callbacks"]
    # train model

    traind_ds = linear_generator(batch_size, input_shape, kernel)
    uft.train(
        traind_ds,
        model,
        loss_fn,
        optimizer,
        epochs,
        batch_size,
        steps_per_epoch=steps_per_epoch,
        callbacks=callback_list,
    )
    # model.fit(
    #     linear_generator(batch_size, input_shape, kernel),
    #     steps_per_epoch=steps_per_epoch,
    #     epochs=epochs,
    #     verbose=1,
    #     callbacks=callback_list,
    # )

    for ll in uft.get_children(model):  # .layers:
        SVmin, SVmax = compute_layer_sv(ll)
        # log metrics
        if SVmin is not None:
            assert SVmax < k_lip_model * kwargs["k_lip_tolerance_factor"], (
                " the maximum singular value of the layer "
                + ll.name
                + " must be lower than the specified boundary"
            )  # noqa: E501

            assert SVmin <= SVmax, (
                " the minimum singular value of the layer "
                + ll.name
                + " must be lower than the maximum value"
            )  # noqa: E501
            if flag_test_SVmin:
                assert SVmin > k_lip_model * (2.0 - kwargs["k_lip_tolerance_factor"]), (
                    " the minimum singular value of the layer "
                    + ll.name
                    + " must be greater than the specified boundary"
                )  # noqa: E501
    return


@pytest.mark.skipif(
    hasattr(compute_layer_sv, "unavailable_class"),
    reason="compute_layer_sv not available",
)
@pytest.mark.parametrize(
    "test_params",
    [
        dict(
            layer_type=SpectralLinear,
            layer_params={"bias": False, "in_features": 4, "out_features": 4},
            batch_size=1000,
            steps_per_epoch=125,
            epochs=5,
            input_shape=(4,),
            k_lip_data=1.0,
            k_lip_model=1.0,
            callbacks=[],
        ),
        dict(
            layer_type=SpectralLinear,
            layer_params={"in_features": 4, "out_features": 4},
            batch_size=1000,
            steps_per_epoch=125,
            epochs=5,
            input_shape=(4,),
            k_lip_data=1.0,
            k_lip_model=5.0,
            callbacks=[],
        ),
        dict(
            layer_type=FrobeniusLinear,
            layer_params={"in_features": 4, "out_features": 1},
            batch_size=1000,
            steps_per_epoch=125,
            epochs=5,
            input_shape=(4,),
            k_lip_data=1.0,
            k_lip_model=1.0,
            test_SVmin=False,
            callbacks=[],
        ),
        dict(
            layer_type=FrobeniusLinear,
            layer_params={"in_features": 4, "out_features": 1},
            batch_size=1000,
            steps_per_epoch=125,
            epochs=5,
            input_shape=(4,),
            k_lip_data=1.0,
            k_lip_model=5.0,
            test_SVmin=False,
            callbacks=[],
        ),
        dict(
            layer_type=SpectralConv2d,
            layer_params={
                "in_channels": 1,
                "out_channels": 2,
                "kernel_size": (3, 3),
                "bias": False,
            },
            batch_size=100,
            steps_per_epoch=125,
            epochs=5,
            input_shape=(1, 5, 5),
            k_lip_data=1.0,
            k_lip_model=1.0,
            k_lip_tolerance_factor=1.02,
            test_SVmin=False,
            callbacks=[],
        ),
        dict(
            layer_type=SpectralConv2d,
            layer_params={
                "in_channels": 1,
                "out_channels": 2,
                "kernel_size": (3, 3),
            },
            batch_size=100,
            steps_per_epoch=125,
            epochs=5,
            input_shape=(1, 5, 5),
            k_lip_data=1.0,
            k_lip_model=5.0,
            k_lip_tolerance_factor=1.02,
            test_SVmin=False,
            callbacks=[],
        ),
        dict(
            layer_type=SpectralConv2d,
            layer_params={
                "in_channels": 3,
                "out_channels": 2,
                "kernel_size": (3, 3),
                "bias": False,
            },
            batch_size=100,
            steps_per_epoch=125,
            epochs=5,
            input_shape=(3, 5, 5),  # case conv_first=False
            k_lip_data=1.0,
            k_lip_model=1.0,
            k_lip_tolerance_factor=1.02,
            test_SVmin=False,
            callbacks=[],
        ),
        dict(
            layer_type=SpectralConv2d,
            layer_params={
                "in_channels": 1,
                "out_channels": 5,
                "kernel_size": (3, 3),
                "bias": False,
                "stride": 2,
            },
            batch_size=100,
            steps_per_epoch=125,
            epochs=5,
            input_shape=(1, 10, 10),
            k_lip_data=1.0,
            k_lip_model=1.0,
            k_lip_tolerance_factor=1.02,
            test_SVmin=False,
            callbacks=[],
        ),
        dict(
            layer_type=SpectralConv2d,
            layer_params={
                "in_channels": 1,
                "out_channels": 3,
                "kernel_size": (3, 3),
                "bias": False,
                "stride": 2,
            },
            batch_size=100,
            steps_per_epoch=125,
            epochs=5,
            input_shape=(1, 10, 10),
            k_lip_data=1.0,
            k_lip_model=1.0,
            k_lip_tolerance_factor=1.02,
            test_SVmin=False,
            callbacks=[],
        ),
        dict(
            layer_type=FrobeniusConv2d,
            layer_params={
                "in_channels": 1,
                "out_channels": 2,
                "kernel_size": (3, 3),
            },
            batch_size=100,
            steps_per_epoch=125,
            epochs=5,
            input_shape=(1, 5, 5),
            k_lip_data=1.0,
            k_lip_model=1.0,
            k_lip_tolerance_factor=1.1,  # Frobenius seems less precise on SVs
            test_SVmin=False,
            callbacks=[],
        ),
        dict(
            layer_type=FrobeniusConv2d,
            layer_params={
                "in_channels": 1,
                "out_channels": 2,
                "kernel_size": (3, 3),
                "bias": False,
            },
            batch_size=100,
            steps_per_epoch=125,
            epochs=5,
            input_shape=(1, 5, 5),
            k_lip_data=1.0,
            k_lip_model=5.0,
            k_lip_tolerance_factor=1.1,
            test_SVmin=False,
            callbacks=[],
        ),
        dict(
            layer_type=FrobeniusConv2d,
            layer_params={
                "in_channels": 3,
                "out_channels": 2,
                "kernel_size": (3, 3),
            },
            batch_size=100,
            steps_per_epoch=125,
            epochs=5,
            input_shape=(3, 5, 5),  # case conv_first=False
            k_lip_data=1.0,
            k_lip_model=1.0,
            k_lip_tolerance_factor=1.1,  # Frobenius seems less precise on SVs
            test_SVmin=False,
            callbacks=[],
        ),
    ],
)
def test_verifySV(test_params):
    train_compute_and_verifySV(**test_params)


@pytest.mark.skipif(
    hasattr(compute_layer_sv, "unavailable_class")
    or hasattr(OrthLinearRegularizer, "unavailable_class"),
    reason="compute_layer_sv or OrthLinearRegularizer not available",
)
@pytest.mark.parametrize(
    "test_params",
    [
        dict(
            layer_type=tLinear,
            layer_params={
                "in_features": 4,
                "out_features": 6,
                "kernel_regularizer": uft.get_instance_framework(
                    OrthLinearRegularizer, {"lambda_orth": 1000.0}
                ),
            },
            batch_size=1000,
            steps_per_epoch=125,
            epochs=10,
            input_shape=(4,),
            k_lip_data=1.0,
            k_lip_model=1.0,
            callbacks=[],
        ),
    ],
)
def test_verifySV_orthRegul(test_params):
    train_compute_and_verifySV(**test_params)
