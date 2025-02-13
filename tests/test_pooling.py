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
import math

import numpy as np
from . import utils_framework as uft
from .utils_framework import (
    CategoricalCrossentropy,
    ScaledAvgPool2d,
    ScaledAdaptiveAvgPool2d,
    ScaledL2NormPool2d,
    ScaledAdaptativeL2NormPool2d,
)


def check_serialization(layer_type, layer_params):
    input_shape = (4, 10, 10)
    input_shape = uft.to_framework_channel(input_shape)
    m = uft.generate_k_lip_model(layer_type, layer_params, input_shape=input_shape, k=1)
    if m is None:
        return
    loss, optimizer, _ = uft.compile_model(
        m,
        optimizer=uft.get_instance_framework(uft.SGD, inst_params={"model": m}),
        loss=CategoricalCrossentropy(from_logits=True),
    )
    name = layer_type.__class__.__name__
    path = os.path.join("logs", "pooling", name)
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


@pytest.mark.parametrize(
    "layer_type",
    [
        ScaledAvgPool2d,
        ScaledL2NormPool2d,
    ],
)
@pytest.mark.parametrize(
    "layer_params",
    [
        {"kernel_size": 2},
        {"kernel_size": (5, 5)},
        {"kernel_size": 2, "k_coef_lip": 2.5},
        {"kernel_size": (2, 2), "stride": (2, 2)},
    ],
)
def test_pooling_simple(layer_type, layer_params):
    check_serialization(layer_type, layer_params)


@pytest.mark.parametrize(
    "layer_type",
    [
        ScaledAdaptiveAvgPool2d,
        ScaledAdaptativeL2NormPool2d,
    ],
)
@pytest.mark.parametrize(
    "layer_params",
    [
        {"output_size": (1, 1)},
        {"output_size": (1, 1), "k_coef_lip": 2.5},
    ],
)
def test_pooling_global(layer_type, layer_params):
    check_serialization(layer_type, layer_params)


@pytest.mark.parametrize(
    "layer_type,layer_params",
    [
        (
            ScaledAvgPool2d,
            {"kernel_size": 2},
        ),
        (
            ScaledAvgPool2d,
            {"kernel_size": (5, 5)},
        ),
        (
            ScaledAvgPool2d,
            {"kernel_size": 2, "k_coef_lip": 2.5},
        ),
        (ScaledAvgPool2d, {"kernel_size": (2, 2), "stride": (2, 2)}),
        (
            ScaledL2NormPool2d,
            {"kernel_size": 2},
        ),
        (
            ScaledL2NormPool2d,
            {"kernel_size": (5, 5)},
        ),
        (
            ScaledL2NormPool2d,
            {"kernel_size": 2, "k_coef_lip": 2.5},
        ),
        (ScaledL2NormPool2d, {"kernel_size": (2, 2), "stride": (2, 2)}),
        (ScaledAdaptiveAvgPool2d, {"output_size": (1, 1)}),
        (ScaledAdaptiveAvgPool2d, {"output_size": (1, 1), "k_coef_lip": 2.5}),
        (ScaledAdaptativeL2NormPool2d, {"output_size": (1, 1)}),
        (ScaledAdaptativeL2NormPool2d, {"output_size": (1, 1), "k_coef_lip": 2.5}),
    ],
)
def test_pool_vanilla_export(layer_type, layer_params):

    input_shape = (4, 10, 10)
    input_shape = uft.to_framework_channel(input_shape)
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


@pytest.mark.parametrize(
    "layer_type, layer_params, expected",
    [
        (
            ScaledAvgPool2d,
            {"kernel_size": 2},
            [
                [
                    [
                        [10.0 / math.sqrt(2.0 * 2.0), 10.0 / math.sqrt(2.0 * 2.0)],
                        [10.0 / math.sqrt(2.0 * 2.0), 10.0 / math.sqrt(2.0 * 2.0)],
                    ],
                    [
                        [10.0 / math.sqrt(2.0 * 2.0), -2.0 / math.sqrt(2.0 * 2.0)],
                        [1.0 / math.sqrt(2.0 * 2.0), 9.0 / math.sqrt(2.0 * 2.0)],
                    ],
                ]
            ],
        ),
        (
            ScaledL2NormPool2d,
            {"kernel_size": 2},
            [
                [
                    [
                        [math.sqrt(30.0), math.sqrt(30.0)],
                        [math.sqrt(30.0), math.sqrt(30.0)],
                    ],
                    [
                        [math.sqrt(74.0), math.sqrt(22.0)],
                        [math.sqrt(129.0), math.sqrt(95.0)],
                    ],
                ]
            ],
        ),
        (
            ScaledAdaptiveAvgPool2d,
            {"output_size": (1, 1)},
            [[[[40.0 / math.sqrt(4.0 * 4.0)]], [[18.0 / math.sqrt(4.0 * 4.0)]]]],
        ),
        (
            ScaledAdaptativeL2NormPool2d,
            {"output_size": (1, 1)},
            [[[[math.sqrt(120.0)]], [[math.sqrt(320.0)]]]],
        ),
    ],
)
def test_AvgPooling(layer_type, layer_params, expected):
    pool = uft.get_instance_framework(layer_type, layer_params)
    if pool is None:
        return
    input = [
        [  # input shape (bc,c,h,w) = (1,2,4,4)
            [
                [1.0, 2.0, 3.0, 4.0],
                [3.0, 4.0, 1.0, 2.0],
                [1.0, 2.0, 3.0, 4.0],
                [3.0, 4.0, 1.0, 2.0],
            ],
            [
                [6.0, 2.0, 1.0, -4.0],
                [5.0, -3.0, -1.0, 2.0],
                [10.0, -2.0, 3.0, 9.0],
                [-3.0, -4.0, -1.0, -2.0],
            ],
        ]
    ]
    xnp = np.asarray(input)
    xnp = uft.to_NCHW_inv(xnp)  # move channel if needed (TF)
    x = uft.to_tensor(xnp)
    uft.build_layer(pool, x.shape[1:])

    y = pool(x).numpy()
    y = np.squeeze(y)  # yorch keep dim whereas tf not
    y_tnp = np.asarray(expected)
    y_t = uft.to_NCHW_inv(y_tnp)  # move channel if needed (TF)
    y_t = np.squeeze(y_t)
    np.testing.assert_almost_equal(y, y_t, decimal=5)
