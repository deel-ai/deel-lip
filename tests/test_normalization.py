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

from .utils_framework import BatchCentering, LayerCentering


def check_serialization(layer_type, layer_params, input_shape=(10,)):
    m = uft.generate_k_lip_model(layer_type, layer_params, input_shape=input_shape, k=1)
    if m is None:
        pytest.skip()
    loss, optimizer, _ = uft.compile_model(
        m,
        optimizer=uft.get_instance_framework(uft.SGD, inst_params={"model": m}),
        loss=uft.CategoricalCrossentropy(from_logits=True),
    )
    name = layer_type.__class__.__name__
    path = os.path.join("logs", "normalization", name)
    xnp = np.random.uniform(-10, 10, (255,) + input_shape)
    x = uft.to_tensor(xnp)
    y1 = m(x)
    uft.save_model(m, path)
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
    hasattr(LayerCentering, "unavailable_class"),
    reason="LayerCentering not available",
)
@pytest.mark.parametrize(
    "size, input_shape, bias",
    [
        (4, (3, 4, 8, 8), False),
        (4, (3, 4, 8, 8), True),
    ],
)
def test_LayerCentering(size, input_shape, bias):
    """evaluate layerbatch centering"""
    input_shape = uft.to_framework_channel(input_shape)
    x = np.arange(np.prod(input_shape)).reshape(input_shape)
    bn = uft.get_instance_framework(LayerCentering, {"size": size, "bias": bias})

    mean_x = np.mean(x, axis=(2, 3))
    mean_shape = (-1, size, 1, 1)
    x = uft.to_tensor(x)
    y = bn(x)
    np.testing.assert_allclose(
        uft.to_numpy(y), x - np.reshape(mean_x, mean_shape), atol=1e-5
    )
    y = bn(2 * x)
    np.testing.assert_allclose(
        uft.to_numpy(y), 2 * x - 2 * np.reshape(mean_x, mean_shape), atol=1e-5
    )  # keep substract batch mean
    bn.eval()
    y = bn(2 * x)
    np.testing.assert_allclose(
        uft.to_numpy(y), 2 * x - 2 * np.reshape(mean_x, mean_shape), atol=1e-5
    )  # eval mode use running_mean


@pytest.mark.skipif(
    hasattr(BatchCentering, "unavailable_class"),
    reason="BatchCentering not available",
)
@pytest.mark.parametrize(
    "size, input_shape, bias",
    [
        (4, (3, 4), False),
        (4, (3, 4), True),
        (4, (3, 4, 8, 8), False),
        (4, (3, 4, 8, 8), True),
    ],
)
def test_BatchCentering(size, input_shape, bias):
    """evaluate layerbatch centering"""
    input_shape = uft.to_framework_channel(input_shape)
    x = np.arange(np.prod(input_shape)).reshape(input_shape)
    bn = uft.get_instance_framework(BatchCentering, {"size": size, "bias": bias})
    bn_mom = bn.momentum
    if len(input_shape) == 2:
        mean_x = np.mean(x, axis=0)
        mean_shape = (1, size)
    else:
        mean_x = np.mean(x, axis=(0, 2, 3))
        mean_shape = (1, size, 1, 1)
    x = uft.to_tensor(x)
    y = bn(x)
    np.testing.assert_allclose(bn.running_mean, mean_x, atol=1e-5)
    np.testing.assert_allclose(
        uft.to_numpy(y), x - np.reshape(mean_x, mean_shape), atol=1e-5
    )
    y = bn(2 * x)
    new_runningmean = mean_x * (1 - bn_mom) + 2 * mean_x * bn_mom
    np.testing.assert_allclose(bn.running_mean, new_runningmean, atol=1e-5)
    np.testing.assert_allclose(
        uft.to_numpy(y), 2 * x - 2 * np.reshape(mean_x, mean_shape), atol=1e-5
    )  # keep substract batch mean
    bn.eval()
    y = bn(2 * x)
    np.testing.assert_allclose(
        bn.running_mean, new_runningmean, atol=1e-5
    )  # eval mode running mean freezed
    np.testing.assert_allclose(
        uft.to_numpy(y), 2 * x - np.reshape(new_runningmean, mean_shape), atol=1e-5
    )  # eval mode use running_mean


@pytest.mark.parametrize(
    "norm_type",
    [LayerCentering, BatchCentering],
)
@pytest.mark.parametrize(
    "size, input_shape, bias",
    [
        (10, (10,), False),
        (10, (10,), True),
        (7, (7, 8, 8), False),
        (7, (7, 8, 8), True),
    ],
)
def test_Normalization_serialization(norm_type, size, input_shape, bias):
    # Check serialization
    if hasattr(norm_type, "unavailable_class"):
        pytest.skip(f"{norm_type} not available")
    check_serialization(
        norm_type, layer_params={"size": size, "bias": bias}, input_shape=input_shape
    )


def linear_generator(batch_size, input_shape: tuple):
    """
    Generate data according to a linear kernel
    Args:
        batch_size: size of each batch
        input_shape: shape of the desired input

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
        batch_y = batch_x
        yield batch_x, batch_y


@pytest.mark.parametrize(
    "norm_type",
    [LayerCentering, BatchCentering],
)
@pytest.mark.parametrize(
    "size, input_shape, bias",
    [
        (10, (10,), True),
        (7, (7, 8, 8), True),
    ],
)
def test_Normalization_bias(norm_type, size, input_shape, bias):
    if hasattr(norm_type, "unavailable_class"):
        pytest.skip(f"{norm_type} not available")
    m = uft.generate_k_lip_model(
        norm_type,
        layer_params={"size": size, "bias": bias},
        input_shape=input_shape,
        k=1,
    )
    if m is None:
        pytest.skip()
    loss, optimizer, _ = uft.compile_model(
        m,
        optimizer=uft.get_instance_framework(uft.SGD, inst_params={"model": m}),
        loss=uft.CategoricalCrossentropy(from_logits=True),
    )
    batch_size = 10
    bb = uft.to_numpy(uft.get_layer_by_index(m, 0).bias)
    np.testing.assert_allclose(bb, np.zeros((size,)), atol=1e-5)

    traind_ds = linear_generator(batch_size, input_shape)
    uft.train(
        traind_ds,
        m,
        loss,
        optimizer,
        2,
        batch_size,
        steps_per_epoch=10,
    )

    bb = uft.to_numpy(uft.get_layer_by_index(m, 0).bias)
    assert np.linalg.norm(bb) != 0.0


@pytest.mark.skipif(
    hasattr(BatchCentering, "unavailable_class"),
    reason="BatchCentering not available",
)
@pytest.mark.parametrize(
    "size, input_shape, bias",
    [
        (4, (3, 4), False),
        (4, (3, 4), True),
        (4, (3, 4, 8, 8), False),
        (4, (3, 4, 8, 8), True),
    ],
)
def test_BatchCentering_runningmean(size, input_shape, bias):
    """evaluate batch centering convergence of running mean"""
    input_shape = uft.to_framework_channel(input_shape)
    # start with 0 to set up running mean to zero
    x = np.zeros(input_shape)
    bn = uft.get_instance_framework(BatchCentering, {"size": size, "bias": bias})
    x = uft.to_tensor(x)
    y = bn(x)

    np.testing.assert_allclose(bn.running_mean, 0.0, atol=1e-5)

    x = np.random.normal(0.0, 1.0, input_shape)
    if len(input_shape) == 2:
        mean_x = np.mean(x, axis=0)
    else:
        mean_x = np.mean(x, axis=(0, 2, 3))
    x = uft.to_tensor(x)
    for _ in range(1000):
        y = bn(x)  # noqa: F841

    np.testing.assert_allclose(bn.running_mean, mean_x, atol=1e-5)
