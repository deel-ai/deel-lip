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
import numpy as np
from . import utils_framework as uft
from .utils_framework import (
    CategoricalCrossentropy,
    GroupSort,
    Householder,
)


def check_serialization(layer_type, layer_params):
    m = uft.generate_k_lip_model(layer_type, layer_params, input_shape=(10,), k=1)
    if m is None:
        return
    optimizer, loss, _ = uft.compile_model(
        m,
        optimizer=uft.get_instance_framework(uft.SGD, inst_params={"model": m}),
        loss=CategoricalCrossentropy(from_logits=True),
    )
    name = layer_type.__class__.__name__
    path = os.path.join("logs", "activations", name)
    xnp = np.random.uniform(-10, 10, (255, 10))
    x = uft.to_tensor(xnp)
    y1 = m(x)
    uft.save_model(m, path)
    m2 = uft.load_model(
        path,
        compile=True,
        layer_type=layer_type,
        layer_params=layer_params,
        input_shape=(10,),
        k=1,
    )
    y2 = m2(x)
    np.testing.assert_allclose(y1.numpy(), y2.numpy())


def test_group_sort_simple():
    check_serialization(GroupSort, layer_params={"group_size": 2})
    check_serialization(GroupSort, layer_params={"group_size": 5})


@pytest.mark.parametrize(
    "group_size,img,expected",
    [
        (
            2,
            False,
            [
                [1.0, 2.0, 3.0, 4.0],
                [3.0, 4.0, 1.0, 2.0],
                [1.0, 2.0, 1.0, 2.0],
                [1.0, 2.0, 1.0, 2.0],
            ],
        ),
        (
            4,
            False,
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
            ],
        ),
        (
            2,
            True,
            [
                [1.0, 2.0, 3.0, 4.0],
                [3.0, 4.0, 1.0, 2.0],
                [1.0, 2.0, 1.0, 2.0],
                [1.0, 2.0, 1.0, 2.0],
            ],
        ),
    ],
)
def test_GroupSort(group_size, img, expected):
    gs = uft.get_instance_framework(GroupSort, {"group_size": group_size})
    if gs is None:
        return
    x = [
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [1, 2, 1, 2],
        [2, 1, 2, 1],
    ]

    if not img:
        x = uft.to_tensor(x)
        uft.build_layer(gs, (4,))
    else:
        xn = np.asarray(x)
        xnp = np.repeat(
            np.expand_dims(np.repeat(np.expand_dims(xn, 1), 28, 1), 1), 28, 1
        )
        x = uft.to_tensor(xnp)
        uft.build_layer(gs, (28, 28, 4))
    y = gs(x).numpy()
    y_t = expected
    if img:
        y_tnp = np.asarray(y_t)
        y_t = np.repeat(
            np.expand_dims(np.repeat(np.expand_dims(y_tnp, 1), 28, 1), 1), 28, 1
        )
    np.testing.assert_equal(y, y_t)


@pytest.mark.parametrize("group_size", [2, 4])
def test_GroupSort_idempotence(group_size):
    gs = uft.get_instance_framework(GroupSort, {"group_size": group_size})
    if gs is None:
        return
    xnp = np.random.uniform(-10, 10, (255, 16))
    x = uft.to_tensor(xnp)
    uft.build_layer(gs, (16,))
    y1 = gs(x)
    y2 = gs(y1)
    np.testing.assert_equal(y1.numpy(), y2.numpy())


"""Tests for Householder activation:
    - instantiation of layer
    - check outputs on dense (bs, n) tensor, with three thetas: 0, pi/2 and pi
    - check outputs on dense (bs, h, w, n) tensor, with three thetas: 0, pi/2 and pi
    - check idempotence hh(hh(x)) = hh(x)
"""


@pytest.mark.skipif(
    hasattr(Householder, "unavailable_class"), reason="Householder not available"
)
@pytest.mark.parametrize(
    "params,shape,len_shape,expected",
    [
        ({}, (28, 28, 10), (5,), np.pi / 2),  # Instantiation without argument
        (
            {
                "data_format": "channels_last",
                "k_coef_lip": 2.5,
                "theta_initializer": "ones",
            },
            (32, 32, 16),
            (8,),
            1,
        ),  # Instantiation with arguments
    ],
)
def test_Householder_instantiation(params, shape, len_shape, expected):
    hh = uft.get_instance_framework(Householder, params)
    uft.build_layer(hh, shape)
    assert hh.theta.shape == len_shape
    np.testing.assert_equal(hh.theta.numpy(), expected)


@pytest.mark.skipif(
    hasattr(Householder, "unavailable_class"), reason="Householder not available"
)
def test_Householder_serialization():
    # Check serialization
    check_serialization(
        Householder, layer_params={"theta_initializer": "glorot_uniform"}
    )

    # Instantiation error because of wrong data format
    with pytest.raises(RuntimeError):
        _ = uft.get_instance_framework(Householder, {"data_format": "channels_first"})


@pytest.mark.skipif(
    hasattr(Householder, "unavailable_class"), reason="Householder not available"
)
@pytest.mark.parametrize("dense", [(True,), (False,)])
def test_Householder_theta_zero(dense):
    """Householder with theta=0 on 2-D tensor (bs, n).
    Theta=0 means Id if z2 > 0, and reflection if z2 < 0.
    """
    hh = uft.get_instance_framework(Householder, {"theta_initializer": "zeros"})
    if dense:
        bs = np.random.randint(64, 512)
        n = np.random.randint(1, 1024) * 2
        size = (bs, n // 2)
    else:  # convolutional
        bs = np.random.randint(32, 128)
        h, w = np.random.randint(1, 64), np.random.randint(1, 64)
        c = np.random.randint(1, 64) * 2
        size = (bs, h, w, c // 2)

    # Case 1: hh(x) = x   (identity case, z2 > 0)
    z1 = np.random.normal(size=size)
    z2 = np.random.uniform(size=size)
    x = np.concatenate([z1, z2], axis=-1)
    np.testing.assert_allclose(hh(uft.to_tensor(x)), x)

    # Case 2: hh(x) = [z1, -z2]   (reflection across z1 axis, z2 < 0)
    z1 = np.random.normal(size=size)
    z2 = -np.random.uniform(size=size)
    x = np.concatenate([z1, z2], axis=-1)
    expected_output = np.concatenate([z1, -z2], axis=-1)
    np.testing.assert_allclose(hh(uft.to_tensor(x)), expected_output)


@pytest.mark.skipif(
    hasattr(Householder, "unavailable_class"), reason="Householder not available"
)
@pytest.mark.parametrize("dense", [(True,), (False,)])
def test_Householder_theta_pi(dense):
    """Householder with theta=pi on 2-D tensor (bs, n).
    Theta=pi means Id if z1 < 0, and reflection if z1 > 0.
    """
    hh = uft.get_instance_framework(
        Householder, {"theta_initializer": uft.initializers_Constant(np.pi)}
    )
    if dense:
        bs = np.random.randint(64, 512)
        n = np.random.randint(1, 1024) * 2
        size = (bs, n // 2)
    else:  # convolutional
        bs = np.random.randint(32, 128)
        h, w = np.random.randint(1, 64), np.random.randint(1, 64)
        c = np.random.randint(1, 64) * 2
        size = (bs, h, w, c // 2)

    # Case 1: hh(x) = x   (identity case, z1 < 0)
    z1 = -np.random.uniform(size=size)
    z2 = np.random.normal(size=size)
    x = np.concatenate([z1, z2], axis=-1)
    np.testing.assert_allclose(hh(uft.to_tensor(x)), x, atol=1e-6)

    # Case 2: hh(x) = [z1, -z2]   (reflection across  z2 axis, z1 > 0)
    z1 = np.random.uniform(size=size)
    z2 = np.random.normal(size=size)
    x = np.concatenate([z1, z2], axis=-1)
    expected_output = np.concatenate([-z1, z2], axis=-1)
    np.testing.assert_allclose(hh(uft.to_tensor(x)), expected_output, atol=1e-6)


@pytest.mark.skipif(
    hasattr(Householder, "unavailable_class"), reason="Householder not available"
)
@pytest.mark.parametrize("dense", [(True,), (False,)])
def test_Householder_theta_90(dense):
    """Householder with theta=pi/2 on 2-D tensor (bs, n).
    Theta=pi/2 is equivalent to GroupSort2: Id if z1 < z2, and reflection if z1 > z2
    """
    hh = uft.get_instance_framework(Householder, {})
    if dense:
        bs = np.random.randint(64, 512)
        n = np.random.randint(1, 1024) * 2
        size = (bs, n // 2)
    else:  # convolutional
        bs = np.random.randint(32, 128)
        h, w = np.random.randint(1, 64), np.random.randint(1, 64)
        c = np.random.randint(1, 64) * 2
        size = (bs, h, w, c // 2)

    # Case 1: hh(x) = x   (identity case, z1 < z2)
    z1 = -np.random.normal(size=size)
    z2 = z1 + np.random.uniform(size=size)
    x = np.concatenate([z1, z2], axis=-1)
    np.testing.assert_allclose(hh(uft.to_tensor(x)), x)

    # Case 2: hh(x) = reflection(x)   (if z1 > z2)
    z1 = np.random.normal(size=size)
    z2 = z1 - np.random.uniform(size=size)
    x = np.concatenate([z1, z2], axis=-1)
    expected_output = np.concatenate([z2, z1], axis=-1)
    np.testing.assert_allclose(hh(uft.to_tensor(x)), expected_output, atol=1e-6)


@pytest.mark.skipif(
    hasattr(Householder, "unavailable_class"), reason="Householder not available"
)
def test_Householder_idempotence():
    """Assert idempotence of Householder activation: hh(hh(x)) = hh(x)"""
    hh = uft.get_instance_framework(
        Householder, {"theta_initializer": "glorot_uniform"}
    )

    bs = np.random.randint(32, 128)
    h, w = np.random.randint(1, 64), np.random.randint(1, 64)
    c = np.random.randint(1, 32) * 2
    x = np.random.normal(size=(bs, h, w, c))
    x = uft.to_tensor(x)

    # Run two times the HH activation and compare both outputs
    y = hh(x)
    z = hh(y)
    np.testing.assert_allclose(y, z)
