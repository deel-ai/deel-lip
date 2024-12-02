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
import pytest
import os
import pprint
import tempfile

import numpy as np


from .utils_framework import (
    SpectralLinear,
    SpectralConv2d,
    SpectralConv1d,
    SpectralConvTranspose2d,
    FrobeniusLinear,
    FrobeniusConv2d,
    ScaledAvgPool2d,
    ScaledAdaptiveAvgPool2d,
    ScaledL2NormPool2d,
    InvertibleDownSampling,
    InvertibleUpSampling,
    ScaledGlobalL2NormPool2d,
    Flatten,
    Sequential,
)
from . import utils_framework as uft


from .utils_framework import (
    tLinear,
    AutoWeightClipConstraint,
    SpectralConstraint,
    FrobeniusConstraint,
    tInput,
    CondenseCallback,
    MonitorCallback,
)

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
- layer structure ( don't check that SpectralConv2d is actually a convolution )

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
    uft.init_session()  # K.clear_session()
    np.random.seed(42)
    input_shape = uft.to_framework_channel(input_shape)
    # create the model, defin opt, and compile it
    model = uft.generate_k_lip_model(layer_type, layer_params, input_shape, k_lip_model)

    optimizer = uft.get_instance_framework(
        uft.Adam, inst_params={"lr": 0.001, "model": model}
    )

    loss_fn, optimizer, metrics = uft.compile_model(
        model,
        optimizer=optimizer,
        loss=uft.MeanSquaredError(),
        metrics=[uft.metric_mse()],
    )
    # model.compile(optimizer=optimizer, loss="mean_squared_error", )
    # create the synthetic data generator
    output_shape = uft.compute_output_shape(input_shape, model)
    kernel = build_kernel(input_shape, output_shape, k_lip_data)
    # define logging features
    logdir = os.path.join("logs", uft.LIP_LAYERS, "%s" % layer_type.__name__)
    os.makedirs(logdir, exist_ok=True)

    callback_list = []
    if kwargs["callbacks"] is not None:
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
        steps_per_epoch=10,
    )
    # the seed is set to compare all models with the same data
    test_dl = linear_generator(batch_size, input_shape, kernel)
    np.random.seed(42)
<<<<<<< HEAD
    set_seed(42)
    loss, mse = model.__getattribute__(EVALUATE)(
        linear_generator(batch_size, input_shape, kernel),
        steps=10,
    )
    empirical_lip_const = evaluate_lip_const(model=model, x=x)
=======
    uft.set_seed(42)

    loss, mse = uft.run_test(model, test_dl, loss_fn, metrics, steps=10)
    x, y = test_dl.send(None)

    x = uft.to_tensor(x)
    empirical_lip_const = uft.evaluate_lip_const(model=model, x=x, seed=42)
>>>>>>> 9e20bf0 (pytest compatible with torchlip)
    # save the model
    model_checkpoint_path = os.path.join(logdir, uft.MODEL_PATH)
    uft.save_model(model, model_checkpoint_path, overwrite=True)
    # model.save(model_checkpoint_path, overwrite=True)
    del model
    uft.init_session()  # K.clear_session()
    model = uft.load_model(
        model_checkpoint_path,
        layer_type=layer_type,
        layer_params=layer_params,
        input_shape=input_shape,
        k=k_lip_model,
    )
<<<<<<< HEAD
    from_empirical_lip_const = evaluate_lip_const(model=model, x=x)
=======
    np.random.seed(42)
    uft.set_seed(42)
    test_dl = linear_generator(batch_size, input_shape, kernel)  # .send(None)
    from_disk_loss, from_disk_mse = uft.run_test(
        model, test_dl, loss_fn, metrics, steps=10
    )
    x, y = test_dl.send(None)
    x = uft.to_tensor(x)
    from_empirical_lip_const = uft.evaluate_lip_const(model=model, x=x, seed=42)

>>>>>>> 9e20bf0 (pytest compatible with torchlip)
    # log metrics
    return (
        mse,
        uft.to_numpy(empirical_lip_const),
        from_disk_mse,
        uft.to_numpy(from_empirical_lip_const),
    )


def _check_mse_results(mse, from_disk_mse, test_params):
    assert from_disk_mse == pytest.approx(
        mse, 1e-5
    ), "serialization must not change the performance of a layer"


def _check_emp_lip_const(emp_lip_const, from_disk_emp_lip_const, test_params):
    assert from_disk_emp_lip_const == pytest.approx(
        emp_lip_const, 1e-5
    ), "serialization must not change the Lipschitz constant of a layer"
    assert (
        emp_lip_const <= test_params["k_lip_model"] * 1.02
    ), " the lip const of the network must be lower than the specified boundary"


def _apply_tests_bank(test_params):
    pp.pprint(test_params)
    (
        mse,
        emp_lip_const,
        from_disk_mse,
        from_disk_emp_lip_const,
    ) = train_k_lip_model(**test_params)
    print("test mse: %f" % mse)
    print(
        "empirical lip const: %f ( expected min data and model %s )"
        % (
            emp_lip_const,
            min(test_params["k_lip_model"], test_params["k_lip_data"]),
        )
    )
    _check_mse_results(mse, from_disk_mse, test_params)
    _check_emp_lip_const(emp_lip_const, from_disk_emp_lip_const, test_params)


@pytest.mark.skipif(
    hasattr(AutoWeightClipConstraint, "unavailable_class"),
    reason="AutoWeightClipConstraint not available",
)
@pytest.mark.parametrize(
    "test_params",
    [
        dict(
            layer_type=tLinear,
            layer_params={
                "in_features": 4,
                "out_features": 4,
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
            layer_type=tLinear,
            layer_params={
                "in_features": 4,
                "out_features": 4,
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
            layer_type=tLinear,
            layer_params={
                "in_features": 4,
                "out_features": 4,
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
    ],
)
def test_constraints_clipping(test_params):
    """
    Tests for a standard Linear layer, for result comparison.
    """
    _apply_tests_bank(test_params)


@pytest.mark.skipif(
    hasattr(SpectralConstraint, "unavailable_class"),
    reason="SpectralConstraint not available",
)
@pytest.mark.parametrize(
    "test_params",
    [
        dict(
            layer_type=tLinear,
            layer_params={
                "in_features": 4,
                "out_features": 4,
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
            layer_type=tLinear,
            layer_params={
                "in_features": 4,
                "out_features": 4,
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
            layer_type=tLinear,
            layer_params={
                "in_features": 4,
                "out_features": 4,
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
    ],
)
def test_constraints_orthogonal(test_params):
    """
    Tests for a standard Linear layer, for result comparison.
    """
    _apply_tests_bank(test_params)


@pytest.mark.skipif(
    hasattr(FrobeniusConstraint, "unavailable_class"),
    reason="FrobeniusConstraint not available",
)
@pytest.mark.parametrize(
    "test_params",
    [
        dict(
            layer_type=tLinear,
            layer_params={
                "in_features": 4,
                "out_features": 4,
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
            layer_type=tLinear,
            layer_params={
                "in_features": 4,
                "out_features": 4,
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
    ],
)
def test_constraints_frobenius(test_params):
    """
    Tests for a standard Linear layer, for result comparison.
    """
    _apply_tests_bank(test_params)


@pytest.mark.parametrize(
    "layer_type",
    [
        SpectralLinear,
    ],
)
@pytest.mark.parametrize(
    "layer_params,k_lip_data,k_lip_model",
    [
        (
            {"bias": False, "in_features": 4, "out_features": 3},
            1.0,
            1.0,
        ),
        (
            {"in_features": 4, "out_features": 4},
            5.0,
            1.0,
        ),
        (
            {"in_features": 4, "out_features": 4},
            1.0,
            5.0,
        ),
    ],
)
def test_spectral_linear(layer_type, layer_params, k_lip_data, k_lip_model):
    test_params = dict(
        layer_type=layer_type,
        layer_params=layer_params,
        batch_size=250,
        steps_per_epoch=125,
        epochs=5,
        input_shape=(4,),
        k_lip_data=k_lip_data,
        k_lip_model=k_lip_model,
        callbacks=[],
    )
    _apply_tests_bank(test_params)


@pytest.mark.parametrize(
    "layer_type",
    [
        FrobeniusLinear,
    ],
)
@pytest.mark.parametrize(
    "layer_params,k_lip_data,k_lip_model",
    [
        (
            {"bias": False, "in_features": 4, "out_features": 1},
            1.0,
            1.0,
        ),
        (
            {"in_features": 4, "out_features": 1},
            5.0,
            1.0,
        ),
        (
            {"in_features": 4, "out_features": 1},
            1.0,
            5.0,
        ),
    ],
)
def test_frobenius_linear(layer_type, layer_params, k_lip_data, k_lip_model):
    test_params = dict(
        layer_type=layer_type,
        layer_params=layer_params,
        batch_size=250,
        steps_per_epoch=125,
        epochs=5,
        input_shape=(4,),
        k_lip_data=k_lip_data,
        k_lip_model=k_lip_model,
        callbacks=[],
    )
    _apply_tests_bank(test_params)


@pytest.mark.parametrize(
    "layer_type",
    [SpectralConv2d, FrobeniusConv2d, SpectralConvTranspose2d],
)
@pytest.mark.parametrize(
    "layer_params,k_lip_data,k_lip_model",
    [
        (
            {
                "in_channels": 1,
                "out_channels": 2,
                "kernel_size": (3, 3),
                "bias": False,
            },
            1.0,
            1.0,
        ),
        (
            {"in_channels": 1, "out_channels": 2, "kernel_size": (3, 3)},
            5.0,
            1.0,
        ),
        (
            {"in_channels": 1, "out_channels": 2, "kernel_size": (3, 3)},
            1.0,
            5.0,
        ),
    ],
)
def test_conv2d(layer_type, layer_params, k_lip_data, k_lip_model):
    if hasattr(layer_type, "unavailable_class"):
        pytest.skip("layer not available")
    test_params = dict(
        layer_type=layer_type,
        layer_params=layer_params,
        batch_size=250,
        steps_per_epoch=125,
        epochs=5,
        input_shape=(1, 5, 5),
        k_lip_data=k_lip_data,
        k_lip_model=k_lip_model,
        callbacks=[],
    )
    _apply_tests_bank(test_params)


@pytest.mark.parametrize(
    "pad_mode",
    [
        "zeros",
        "reflect",
        "circular",
        "symmetric",
    ],
)
@pytest.mark.parametrize(
    "pad, kernel_size",
    [
        (1, (3, 3)),
        ((1, 1), (3, 3)),
        (2, (5, 5)),
        ((2, 2), (5, 5)),
    ],
)
@pytest.mark.parametrize(
    "layer_params,k_lip_data,k_lip_model",
    [
        (
            {
                "in_channels": 1,
                "out_channels": 2,
                "bias": False,
            },
            1.0,
            1.0,
        ),
        (
            {"in_channels": 1, "out_channels": 2},
            1.0,
            1.0,
        ),
        (
            {"in_channels": 1, "out_channels": 2},
            1.0,
            5.0,
        ),
    ],
)
def test_spectralconv2d_pad(
    pad, pad_mode, kernel_size, layer_params, k_lip_data, k_lip_model
):
    layer_params["padding"] = pad
    layer_params["padding_mode"] = pad_mode
    layer_params["kernel_size"] = kernel_size
    if not uft.is_supported_padding(pad_mode,SpectralConv2d):
        pytest.skip(f"SpectralConv2d: Padding {pad_mode} not supported")
    test_params = dict(
        layer_type=SpectralConv2d,
        layer_params=layer_params,
        batch_size=250,
        steps_per_epoch=125,
        epochs=5,
        input_shape=(1, 5, 5),
        k_lip_data=k_lip_data,
        k_lip_model=k_lip_model,
        callbacks=[],
    )
    _apply_tests_bank(test_params)


@pytest.mark.skipif(
    hasattr(SpectralConv1d, "unavailable_class"),
    reason="SpectralConv1d not available",
)
@pytest.mark.parametrize(
    "test_params",
    [
        dict(
            layer_type=SpectralConv1d,
            layer_params={
                "in_channels": 1,
                "out_channels": 2,
                "kernel_size": 3,
                "bias": False,
            },
            batch_size=250,
            steps_per_epoch=125,
            epochs=5,
            input_shape=(1, 5),
            k_lip_data=1.0,
            k_lip_model=1.0,
            callbacks=[],
        ),
        dict(
            layer_type=SpectralConv1d,
            layer_params={"in_channels": 1, "out_channels": 2, "kernel_size": 3},
            batch_size=250,
            steps_per_epoch=125,
            epochs=5,
            input_shape=(1, 5),
            k_lip_data=5.0,
            k_lip_model=1.0,
            callbacks=[],
        ),
        dict(
            layer_type=SpectralConv1d,
            layer_params={"in_channels": 1, "out_channels": 2, "kernel_size": 3},
            batch_size=250,
            steps_per_epoch=125,
            epochs=5,
            input_shape=(1, 5),
            k_lip_data=1.0,
            k_lip_model=5.0,
            callbacks=[],
        ),
    ],
)
def test_spectralconv1d(test_params):
    _apply_tests_bank(test_params)


@pytest.mark.parametrize(
    "test_params",
    [
        dict(
            layer_type=ScaledAvgPool2d,
            layer_params={"kernel_size": (3, 3)},
            batch_size=250,
            steps_per_epoch=1,
            epochs=1,
            input_shape=(1, 6, 6),
            k_lip_data=1.0,
            k_lip_model=1.0,
            callbacks=[],
        ),
        dict(
            layer_type=ScaledAvgPool2d,
            layer_params={"kernel_size": (3, 3)},
            batch_size=250,
            steps_per_epoch=1,
            epochs=1,
            input_shape=(1, 6, 6),
            k_lip_data=1.0,
            k_lip_model=5.0,
            callbacks=[],
        ),
        dict(
            layer_type=ScaledAvgPool2d,
            layer_params={"kernel_size": (3, 3)},
            batch_size=250,
            steps_per_epoch=1,
            epochs=1,
            input_shape=(1, 6, 6),
            k_lip_data=5.0,
            k_lip_model=1.0,
            callbacks=[],
        ),
    ],
)
def test_ScaledAvgPool2d(test_params):
    # tests only checks that lip cons is enforced
    _apply_tests_bank(test_params)


@pytest.mark.parametrize(
    "test_params",
    [
        # tests only checks that lip cons is enforced
        dict(
            layer_type=Sequential,
            layer_params={
                "layers": [
                    tInput(uft.to_framework_channel((1, 5, 5))),
                    uft.get_instance_framework(
                        ScaledAdaptiveAvgPool2d,
                        {"data_format": "channels_last", "output_size": (1, 1)},
                    ),
                ]
            },
            batch_size=250,
            steps_per_epoch=1,
            epochs=1,
            input_shape=(1, 5, 5),
            k_lip_data=1.0,
            k_lip_model=1.0,
            callbacks=[],
        ),
        dict(
            layer_type=Sequential,
            layer_params={
                "layers": [
                    tInput(uft.to_framework_channel((1, 5, 5))),
                    uft.get_instance_framework(
                        ScaledAdaptiveAvgPool2d,
                        {"data_format": "channels_last", "output_size": (1, 1)},
                    ),
                ]
            },
            batch_size=250,
            steps_per_epoch=1,
            epochs=1,
            input_shape=(1, 5, 5),
            k_lip_data=5.0,
            k_lip_model=1.0,
            callbacks=[],
        ),
        dict(
            layer_type=Sequential,
            layer_params={
                "layers": [
                    tInput(uft.to_framework_channel((1, 5, 5))),
                    uft.get_instance_framework(
                        ScaledAdaptiveAvgPool2d,
                        {"data_format": "channels_last", "output_size": (1, 1)},
                    ),
                ]
            },
            batch_size=250,
            steps_per_epoch=1,
            epochs=1,
            input_shape=(1, 5, 5),
            k_lip_data=1.0,
            k_lip_model=5.0,
            callbacks=[],
        ),
    ],
)
def test_ScaledAdaptiveAvgPool2d(test_params):
    # tests only checks that lip cons is enforced
    _apply_tests_bank(test_params)


@pytest.mark.parametrize(
    "test_params",
    [
        # tests only checks that lip cons is enforced
        dict(
            layer_type=Sequential,
            layer_params={
                "layers": [
                    tInput(uft.to_framework_channel((1, 5, 5))),
                    uft.get_instance_framework(
                        ScaledL2NormPool2d,
                        {"kernel_size": (2, 3), "data_format": "channels_last"},
                    ),
                ]
            },
            batch_size=250,
            steps_per_epoch=1,
            epochs=1,
            input_shape=(1, 5, 5),
            k_lip_data=1.0,
            k_lip_model=1.0,
            callbacks=[],
        ),
        dict(
            layer_type=Sequential,
            layer_params={
                "layers": [
                    tInput(uft.to_framework_channel((1, 5, 5))),
                    uft.get_instance_framework(
                        ScaledL2NormPool2d,
                        {"kernel_size": (2, 3), "data_format": "channels_last"},
                    ),
                ]
            },
            batch_size=250,
            steps_per_epoch=1,
            epochs=1,
            input_shape=(1, 5, 5),
            k_lip_data=5.0,
            k_lip_model=1.0,
            callbacks=[],
        ),
        dict(
            layer_type=Sequential,
            layer_params={
                "layers": [
                    tInput(uft.to_framework_channel((1, 5, 5))),
                    uft.get_instance_framework(
                        ScaledL2NormPool2d,
                        {"kernel_size": (2, 3), "data_format": "channels_last"},
                    ),
                ]
            },
            batch_size=250,
            steps_per_epoch=1,
            epochs=1,
            input_shape=(1, 5, 5),
            k_lip_data=1.0,
            k_lip_model=5.0,
            callbacks=[],
        ),
    ],
)
def test_scaledl2normPool2d(test_params):
    # tests only checks that lip cons is enforced
    _apply_tests_bank(test_params)


@pytest.mark.skipif(
    hasattr(ScaledGlobalL2NormPool2d, "unavailable_class"),
    reason="compute_layer_sv not available",
)
@pytest.mark.parametrize(
    "test_params",
    [
        # tests only checks that lip cons is enforced
        dict(
            layer_type=Sequential,
            layer_params={
                "layers": [
                    tInput(uft.to_framework_channel((1, 5, 5))),
                    uft.get_instance_framework(
                        ScaledGlobalL2NormPool2d, {"data_format": "channels_last"}
                    ),
                ]
            },
            batch_size=250,
            steps_per_epoch=1,
            epochs=1,
            input_shape=(1, 5, 5),
            k_lip_data=1.0,
            k_lip_model=1.0,
            callbacks=[],
        ),
        dict(
            layer_type=Sequential,
            layer_params={
                "layers": [
                    tInput(uft.to_framework_channel((1, 5, 5))),
                    uft.get_instance_framework(
                        ScaledGlobalL2NormPool2d, {"data_format": "channels_last"}
                    ),
                ]
            },
            batch_size=250,
            steps_per_epoch=1,
            epochs=1,
            input_shape=(1, 5, 5),
            k_lip_data=5.0,
            k_lip_model=1.0,
            callbacks=[],
        ),
        dict(
            layer_type=Sequential,
            layer_params={
                "layers": [
                    tInput(uft.to_framework_channel((1, 5, 5))),
                    uft.get_instance_framework(
                        ScaledGlobalL2NormPool2d, {"data_format": "channels_last"}
                    ),
                ]
            },
            batch_size=250,
            steps_per_epoch=1,
            epochs=1,
            input_shape=(1, 5, 5),
            k_lip_data=1.0,
            k_lip_model=5.0,
            callbacks=[],
        ),
    ],
)
def test_scaledgloball2normPool2d(test_params):
    # tests only checks that lip cons is enforced
    _apply_tests_bank(test_params)


@pytest.mark.parametrize(
    "test_params",
    [
        dict(
            layer_type=Sequential,
            layer_params={
                "layers": [
                    tInput(uft.to_framework_channel((1, 5, 5))),
                    uft.get_instance_framework(
                        SpectralConv2d,
                        {
                            "in_channels": 1,
                            "out_channels": 2,
                            "kernel_size": (3, 3),
                            "padding": 1,
                            "bias": False,
                        },
                    ),
                    uft.get_instance_framework(Flatten, {}),
                    uft.get_instance_framework(
                        SpectralLinear, {"in_features": 50, "out_features": 4}
                    ),
                ]
            },
            batch_size=250,
            steps_per_epoch=125,
            epochs=5,
            input_shape=(1, 5, 5),
            k_lip_data=1.0,
            k_lip_model=1.0,
            callbacks=[],
        ),
        dict(
            layer_type=Sequential,
            layer_params={
                "layers": [
                    tInput(uft.to_framework_channel((1, 5, 5))),
                    uft.get_instance_framework(
                        SpectralConv2d,
                        {
                            "in_channels": 1,
                            "out_channels": 2,
                            "padding": 1,
                            "kernel_size": (3, 3),
                        },
                    ),
                    uft.get_instance_framework(Flatten, {}),
                    uft.get_instance_framework(
                        SpectralLinear, {"in_features": 50, "out_features": 4}
                    ),
                ]
            },
            batch_size=250,
            steps_per_epoch=125,
            epochs=5,
            input_shape=(1, 5, 5),
            k_lip_data=5.0,
            k_lip_model=1.0,
            callbacks=[],
        ),
        dict(
            layer_type=Sequential,
            layer_params={
                "layers": [
                    tInput(uft.to_framework_channel((1, 5, 5))),
                    uft.get_instance_framework(
                        SpectralConv2d,
                        {
                            "in_channels": 1,
                            "out_channels": 2,
                            "padding": 1,
                            "kernel_size": (3, 3),
                        },
                    ),
                    uft.get_instance_framework(Flatten, {}),
                    uft.get_instance_framework(
                        SpectralLinear, {"in_features": 50, "out_features": 4}
                    ),
                ]
            },
            batch_size=250,
            steps_per_epoch=125,
            epochs=5,
            input_shape=(1, 5, 5),
            k_lip_data=1.0,
            k_lip_model=5.0,
            callbacks=[],
        ),
    ],
)
def test_multilayer(test_params):
    _apply_tests_bank(test_params)


def build_test_callbacks():
    list_tests = []
    if not hasattr(CondenseCallback, "unavailable_class"):
        list_tests.append(
            dict(
                layer_type=Sequential,
                layer_params={
                    "layers": [
                        tInput(uft.to_framework_channel((1, 5, 5))),
                        uft.get_instance_framework(
                            SpectralConv2d,
                            {
                                "in_channels": 1,
                                "out_channels": 2,
                                "padding": 1,
                                "kernel_size": (3, 3),
                            },
                        ),
                        uft.get_instance_framework(
                            ScaledAvgPool2d, {"kernel_size": (2, 2)}
                        ),
                        uft.get_instance_framework(Flatten, {}),
                        uft.get_instance_framework(
                            SpectralLinear, {"in_features": 8, "out_features": 4}
                        ),
                    ]
                },
                batch_size=250,
                steps_per_epoch=125,
                epochs=5,
                input_shape=(1, 5, 5),
                k_lip_data=1.0,
                k_lip_model=1.0,
                callbacks=[CondenseCallback(on_batch=True, on_epoch=False)],
            )
        )

    if not hasattr(MonitorCallback, "unavailable_class"):
        list_tests.append(
            dict(
                layer_type=Sequential,
                layer_params={
                    "layers": [
                        tInput(uft.to_framework_channel((1, 5, 5))),
                        uft.get_instance_framework(
                            SpectralConv2d,
                            {
                                "in_channels": 1,
                                "out_channels": 2,
                                "kernel_size": (3, 3),
                                "padding": 1,
                                "name": "conv1",
                            },
                        ),
                        uft.get_instance_framework(
                            ScaledAvgPool2d, {"kernel_size": (2, 2)}
                        ),
                        uft.get_instance_framework(Flatten, {}),
                        uft.get_instance_framework(
                            SpectralLinear,
                            {"name": "dense1", "in_features": 8, "out_features": 4},
                        ),
                    ]
                },
                batch_size=250,
                steps_per_epoch=125,
                epochs=5,
                input_shape=(1, 5, 5),
                k_lip_data=1.0,
                k_lip_model=1.0,
                callbacks=[
                    # CondenseCallback(on_batch=False, on_epoch=True),
                    MonitorCallback(
                        monitored_layers=["conv1", "dense1"],
                        logdir=os.path.join("logs", uft.LIP_LAYERS, "Sequential"),
                        target="kernel",
                        what="all",
                        on_epoch=False,
                        on_batch=True,
                    ),
                    MonitorCallback(
                        monitored_layers=["conv1", "dense1"],
                        logdir=os.path.join("logs", uft.LIP_LAYERS, "Sequential"),
                        target="wbar",
                        what="all",
                        on_epoch=False,
                        on_batch=True,
                    ),
                ],
            )
        )
    return list_tests


@pytest.mark.parametrize("test_params", build_test_callbacks())
def test_callbacks(test_params):
    _apply_tests_bank(test_params)


@pytest.mark.parametrize(
    "test_params",
    [
        dict(
            layer_type=InvertibleDownSampling,
            layer_params={"kernel_size": 3},
            batch_size=250,
            steps_per_epoch=1,
            epochs=5,
            input_shape=(3, 6, 6),
            k_lip_data=1.0,
            k_lip_model=1.0,
            callbacks=[],
        ),
    ],
)
def test_invertibledownsampling(test_params):
    # tests only checks that lip cons is enforced
    _apply_tests_bank(test_params)


@pytest.mark.parametrize(
    "test_params",
    [
        dict(
            layer_type=InvertibleUpSampling,
            layer_params={"kernel_size": 3},
            batch_size=250,
            steps_per_epoch=1,
            epochs=5,
            input_shape=(18, 6, 6),
            k_lip_data=1.0,
            k_lip_model=1.0,
            callbacks=[],
        ),
    ],
)
def test_invertibleupsampling(test_params):
    # tests only checks that lip cons is enforced
    _apply_tests_bank(test_params)


@pytest.mark.skipif(
    hasattr(SpectralConvTranspose2d, "unavailable_class"),
    reason="SpectralConvTranspose2d not available",
)
@pytest.mark.parametrize(
    "test_params,msg",
    [
        (dict(in_channels=1, out_channels=5, kernel_size=3), ""),
        (
            dict(in_channels=1, out_channels=12, kernel_size=5, stride=2, bias=False),
            "",
        ),
        (
            dict(
                in_channels=1,
                out_channels=3,
                kernel_size=3,
                padding="same",
                dilation=1,
            ),
            "",
        ),
        (
            dict(
                in_channels=1,
                out_channels=4,
                kernel_size=1,
                output_padding=None,
                activation="relu",
            ),
            "",
        ),
        (
            dict(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                data_format="channels_first",
            ),
            "",
        ),
        (
            dict(
                in_channels=1,
                out_channels=10,
                kernel_size=3,
                padding=0,
                padding_mode="valid",
            ),
            "Wrong padding",
        ),
        (
            dict(in_channels=1, out_channels=10, kernel_size=3, dilation=2),
            "Wrong dilation rate",
        ),
        (
            dict(in_channels=1, out_channels=10, kernel_size=3, output_padding=5),
            "Wrong data format",
        ),
    ],
)
def test_SpectralConvTranspose2d_instantiation(test_params, msg):
    if msg == "":
        uft.get_instance_framework(SpectralConvTranspose2d, test_params)
    else:
        with pytest.raises(ValueError):
            uft.get_instance_framework(SpectralConvTranspose2d, test_params)


@pytest.mark.skipif(
    hasattr(SpectralConv1d, "unavailable_class"),
    reason="SpectralConv1d not available",
)
@pytest.mark.parametrize(
    "pad_mode",
    [
        "zeros",
        "reflect",
        "circular",
        "symmetric",
    ],
)
@pytest.mark.parametrize(
    "pad, kernel_size",
    [
        (1, (3,)),
        (2, (5,)),
    ],
)
@pytest.mark.parametrize(
    "layer_type",
    [
        SpectralConv1d,
    ],
)
@pytest.mark.parametrize(
    "layer_params",
    [
        {
            "in_channels": 1,
            "out_channels": 2,
            "bias": False,
        },
        {"in_channels": 1, "out_channels": 2},
    ],
)
def test_SpectralConv1d_vanilla_export(
    pad, pad_mode, kernel_size, layer_params, layer_type
):
    layer_params["padding"] = pad
    layer_params["padding_mode"] = pad_mode
    layer_params["kernel_size"] = kernel_size
    layer_type = layer_type
    input_shape = (1, 5)

    model = uft.generate_k_lip_model(layer_type, layer_params, input_shape, 1.0)

    # lay = SpectralConvTranspose2d(**kwargs)
    # model = Sequential([lay])
    x = np.random.normal(size=(5,) + input_shape)

    x = uft.to_tensor(x)
    y1 = model(x)

    # Test vanilla export inference comparison
    if uft.vanilla_require_a_copy():
        model2 = uft.generate_k_lip_model(layer_type, layer_params, input_shape, 1.0)
        uft.copy_model_parameters(model, model2)
        vanilla_model = uft.vanillaModel(model2)
    else:
        vanilla_model = uft.vanillaModel(model)  # .vanilla_export()
    y2 = vanilla_model(x)
    np.testing.assert_allclose(uft.to_numpy(y1), uft.to_numpy(y2), atol=1e-6)

    # Test saving/loading model
    with tempfile.TemporaryDirectory() as tmpdir:
        uft.MODEL_PATH = os.path.join(tmpdir, uft.MODEL_PATH)
        uft.save_model(model, uft.MODEL_PATH, overwrite=True)
        uft.load_model(
            uft.MODEL_PATH,
            layer_type=layer_type,
            layer_params=layer_params,
            input_shape=input_shape,
            k=1.0,
        )


@pytest.mark.skipif(
    hasattr(SpectralConv2d, "unavailable_class"),
    reason="SpectralConv2d not available",
)
@pytest.mark.parametrize(
    "pad_mode",
    [
        "zeros",
        "reflect",
        "circular",
        "symmetric",
    ],
)
@pytest.mark.parametrize(
    "pad, kernel_size",
    [
        (1, (3, 3)),
        ((1, 1), (3, 3)),
        (2, (5, 5)),
        ((2, 2), (5, 5)),
    ],
)
@pytest.mark.parametrize(
    "layer_type",
    [
        SpectralConv2d,
        FrobeniusConv2d,
    ],
)
@pytest.mark.parametrize(
    "layer_params",
    [
        {
            "in_channels": 1,
            "out_channels": 2,
            "bias": False,
        },
        {"in_channels": 1, "out_channels": 2},
    ],
)
def test_Conv2d_vanilla_export(pad, pad_mode, kernel_size, layer_params, layer_type):
    layer_params["padding"] = pad
    layer_params["padding_mode"] = pad_mode
    layer_params["kernel_size"] = kernel_size
    layer_type = layer_type
    input_shape = (1, 5, 5)
    
    if not uft.is_supported_padding(pad_mode,layer_type):
        pytest.skip(f"{layer_type}: Padding {pad_mode} not supported")
    model = uft.generate_k_lip_model(layer_type, layer_params, input_shape, 1.0)

    # lay = SpectralConvTranspose2d(**kwargs)
    # model = Sequential([lay])
    x = np.random.normal(size=(5,) + input_shape)

    x = uft.to_tensor(x)
    y1 = model(x)

    # Test vanilla export inference comparison
    if uft.vanilla_require_a_copy():
        model2 = uft.generate_k_lip_model(layer_type, layer_params, input_shape, 1.0)
        uft.copy_model_parameters(model, model2)
        vanilla_model = uft.vanillaModel(model2)
    else:
        vanilla_model = uft.vanillaModel(model)  # .vanilla_export()
    y2 = vanilla_model(x)
    np.testing.assert_allclose(uft.to_numpy(y1), uft.to_numpy(y2), atol=1e-6)

    # Test saving/loading model
    with tempfile.TemporaryDirectory() as tmpdir:
        uft.MODEL_PATH = os.path.join(tmpdir, uft.MODEL_PATH)
        uft.save_model(model, uft.MODEL_PATH, overwrite=True)
        uft.load_model(
            uft.MODEL_PATH,
            layer_type=layer_type,
            layer_params=layer_params,
            input_shape=input_shape,
            k=1.0,
        )


@pytest.mark.skipif(
    hasattr(SpectralConvTranspose2d, "unavailable_class"),
    reason="SpectralConvTranspose2d not available",
)
def test_SpectralConvTranspose2d_vanilla_export():
    kwargs = dict(
        in_channels=3,
        out_channels=16,
        kernel_size=5,
        stride=2,
        activation="relu",
        data_format="channels_first",
        input_shape=(3, 28, 28),
    )

    model = uft.generate_k_lip_model(
        SpectralConvTranspose2d, kwargs, kwargs["input_shape"], 1.0
    )

    # lay = SpectralConvTranspose2d(**kwargs)
    # model = Sequential([lay])
    x = np.random.normal(size=(5,) + kwargs["input_shape"])

    x = uft.to_tensor(x)
    y1 = model(x)

    # Test vanilla export inference comparison
    if uft.vanilla_require_a_copy():
        model2 = uft.generate_k_lip_model(
            SpectralConvTranspose2d, kwargs, kwargs["input_shape"], 1.0
        )
        uft.copy_model_parameters(model, model2)
        vanilla_model = uft.vanillaModel(model2)
    else:
        vanilla_model = uft.vanillaModel(model)  # .vanilla_export()
    y2 = vanilla_model(x)
    np.testing.assert_allclose(uft.to_numpy(y1), uft.to_numpy(y2), atol=1e-6)

    # Test saving/loading model
    with tempfile.TemporaryDirectory() as tmpdir:
        uft.MODEL_PATH = os.path.join(tmpdir, uft.MODEL_PATH)
        uft.save_model(model, uft.MODEL_PATH, overwrite=True)
        uft.load_model(
            uft.MODEL_PATH,
            layer_type=SpectralConvTranspose2d,
            layer_params=kwargs,
            input_shape=kwargs["input_shape"],
            k=1.0,
        )
