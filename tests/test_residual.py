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
import os
import pytest

import numpy as np

from . import utils_framework as uft

from .utils_framework import LipResidual
from .utils_framework import tInput, tSplit, tModel


def get_functional_tensors(input_shape):
    dict_functional_tensors = {}
    dict_functional_tensors["inputs"] = uft.get_instance_framework(
        tInput, {"shape": input_shape}
    )
    dict_functional_tensors["split"] = uft.get_instance_framework(
        tSplit, {"chunks": 2, "dim": 1}
    )
    dict_functional_tensors["residual"] = uft.get_instance_framework(LipResidual, {})
    return dict_functional_tensors


def functional_input_output_tensors(dict_functional_tensors, x):
    """Return input and output tensor of a Functional (hard-coded) model"""
    if dict_functional_tensors["inputs"] is None:
        inputs = x
    else:
        inputs = dict_functional_tensors["inputs"]
    x = dict_functional_tensors["split"](inputs)
    outputs = dict_functional_tensors["residual"](x[0], x[1])
    if dict_functional_tensors["inputs"] is None:
        return outputs
    else:
        return inputs, outputs
    # return x


def check_serialization(layer_type, layer_params, input_shape=(10,)):

    dict_tensors = get_functional_tensors(input_shape)
    m = uft.get_functional_model(tModel, dict_tensors, functional_input_output_tensors)
    if m is None:
        pytest.skip()
    loss, optimizer, _ = uft.compile_model(
        m,
        optimizer=uft.get_instance_framework(uft.SGD, inst_params={"model": m}),
        loss=uft.MeanSquaredError(),
    )
    name = layer_type.__class__.__name__
    path = os.path.join("logs", "residual", name)
    xnp = np.random.uniform(-10, 10, (255,) + input_shape)
    x = uft.to_tensor(xnp)
    y1 = m(x)
    uft.save_model(m, path)

    # build and generate the model
    if uft.vanilla_require_a_copy():
        dict_tensors2 = get_functional_tensors(input_shape)
        m2 = uft.get_functional_model(
            tModel, dict_tensors2, functional_input_output_tensors
        )
        m2 = uft.load_state_dict(path, m2)
    else:
        m2 = uft.load_model(
            path,
            compile=True,
            layer_type=layer_type,
            layer_params=layer_params,
            input_shape=input_shape,
            k=1,
        )
    y2 = m2(x)
    np.testing.assert_allclose(uft.to_numpy(y1), uft.to_numpy(y2))


@pytest.mark.skipif(
    hasattr(LipResidual, "unavailable_class"),
    reason="LipResidual not available",
)
@pytest.mark.parametrize(
    "input_shape",
    [
        (3, 4, 8, 8),
    ],
)
def test_initLipResidual(input_shape):
    """evaluate layerbatch centering"""
    input_shape = uft.to_framework_channel(input_shape)
    x1 = np.arange(np.prod(input_shape)).reshape(input_shape)
    x2 = np.zeros(input_shape)
    res = uft.get_instance_framework(LipResidual, {})

    alpha_res = uft.to_numpy(res.alpha)
    assert alpha_res == 0.0
    z = res(uft.to_tensor(x1), uft.to_tensor(x1))
    np.testing.assert_allclose(uft.to_numpy(z), x1, atol=1e-5)
    z = res(uft.to_tensor(x1), uft.to_tensor(x2))
    np.testing.assert_allclose(uft.to_numpy(z), x1 / 2.0, atol=1e-5)


@pytest.mark.skipif(
    hasattr(LipResidual, "unavailable_class"),
    reason="LipResidual not available",
)
@pytest.mark.parametrize(
    "input_shape",
    [
        (14, 8, 8),
    ],
)
def test_Normalization_serialization(input_shape):
    # Check serialization
    check_serialization(LipResidual, layer_params={}, input_shape=input_shape)


def linear_generator(batch_size, input_shape: tuple, input_type: str):
    """
    Generate data according to a linear kernel
    Args:
        batch_size: size of each batch
        input_shape: shape of the desired input
        input_type: duplication type for residual

    Returns:
        a generator for the data

    """
    input_shape = tuple(
        [sh // 2 if id == 0 else sh for id, sh in enumerate(input_shape)]
    )
    while True:
        # pick random sample in [0, 1] with the input shape
        batch_x = np.array(
            np.random.uniform(-10, 10, (batch_size,) + input_shape), dtype=np.float16
        )
        # same output as input
        batch_y = batch_x
        # concatenate to use split
        if input_type == "zeros":
            batch_x = np.concatenate([batch_x, np.zeros_like(batch_x)], axis=1)
        if input_type == "invert":
            batch_x = np.concatenate([np.zeros_like(batch_x), batch_x], axis=1)
        if input_type == "copy":
            batch_x = np.concatenate([batch_x, batch_x], axis=1)
        if input_type == "random":
            batch_x = np.concatenate(
                [
                    batch_x,
                    np.array(
                        np.random.uniform(-10, 10, (batch_size,) + input_shape),
                        dtype=np.float16,
                    ),
                ],
                axis=1,
            )
        yield batch_x, batch_y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


@pytest.mark.skipif(
    hasattr(LipResidual, "unavailable_class"),
    reason="LipResidual not available",
)
@pytest.mark.parametrize(
    "input_shape, input_type, learnt_alpha",
    [
        ((14, 8, 8), "zeros", 1.0),  # x1=x x2=0
        ((14, 8, 8), "copy", 0.5),  # x1=x2=x
        ((14, 8, 8), "invert", 0.0),  # x1=0 x2=x
        ((14, 8, 8), "random", None),  # x1=x1 x2=x2
    ],
)
def test_learntResidual(input_shape, input_type, learnt_alpha):
    dict_tensors = get_functional_tensors(input_shape)
    m = uft.get_functional_model(tModel, dict_tensors, functional_input_output_tensors)
    if m is None:
        pytest.skip()
    loss, optimizer, _ = uft.compile_model(
        m,
        optimizer=uft.get_instance_framework(uft.SGD, inst_params={"model": m}),
        loss=uft.MeanSquaredError(),
    )
    batch_size = 10

    traind_ds = linear_generator(batch_size, input_shape, input_type)
    uft.train(
        traind_ds,
        m,
        loss,
        optimizer,
        5,
        batch_size,
        steps_per_epoch=100,
    )

    alpha = uft.to_numpy(m.get_module_by_name("residual").alpha)
    if learnt_alpha is not None:
        np.testing.assert_allclose(sigmoid(alpha), learnt_alpha, atol=1e-1)
    else:
        assert alpha != 0.0
