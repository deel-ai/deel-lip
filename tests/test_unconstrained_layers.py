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

import numpy as np

from . import utils_framework as uft

from .utils_framework import pad_input, PadConv2d
from .utils_framework import tConv2d
from .utils_framework import vanillaModel


def compare(x, x_ref, index_x=[], index_x_ref=[]):
    """Compare a tensor and its padded version, based on index_x and ref."""
    x = uft.to_NCHW(uft.to_numpy(x))
    x_ref = uft.to_NCHW(uft.to_numpy(x_ref))
    x_cropped = x[:, :, index_x[0] : index_x[1], index_x[3] : index_x[4]][
        :, :, :: index_x[2], :: index_x[5]
    ]
    if index_x_ref[0] is None:  # compare with 0
        np.testing.assert_allclose(x_cropped, np.zeros(x_cropped.shape), 1e-2, 0)
    else:
        np.testing.assert_allclose(
            x_cropped
            - x_ref[
                :, :, index_x_ref[0] : index_x_ref[1], index_x_ref[3] : index_x_ref[4]
            ][:, :, :: index_x_ref[2], :: index_x_ref[5]],
            np.zeros(x_cropped.shape),
            1e-2,
            0,
        )


@pytest.mark.parametrize(
    "padding_tested", ["circular", "constant", "symmetric", "reflect", "replicate"]
)
@pytest.mark.parametrize(
    "input_shape, batch_size, kernel_size, filters",
    [
        ((1, 5, 5), 250, (3, 3), 2),
        ((5, 5, 5), 250, (3, 3), 2),
        ((2, 7, 7), 250, (7, 7), 2),
        ((128, 5, 5), 250, 3, 2),
    ],
)
def test_padding(padding_tested, input_shape, batch_size, kernel_size, filters):
    """Test different padding types: assert values in original and padded tensors"""
    input_shape = uft.to_framework_channel(input_shape)
    if not uft.is_supported_padding(padding_tested,PadConv2d):
        pytest.skip(f"Padding {padding_tested} not supported")
    kernel_size_list = kernel_size
    if isinstance(kernel_size, (int, float)):
        kernel_size_list = [kernel_size, kernel_size]

    x = np.random.normal(size=(batch_size,) + input_shape).astype("float32")
    x = uft.to_tensor(x)
    x_pad = pad_input(x, padding_tested, kernel_size)
    p_vert, p_hor = kernel_size_list[0] // 2, kernel_size_list[1] // 2
    x_pad_NCHW = uft.get_NCHW(x_pad)
    x_NCHW = uft.get_NCHW(x)

    center_x_pad = [p_vert, -p_vert, 1, p_hor, -p_hor, 1, "center"]
    upper_x_pad = [0, p_vert, 1, p_hor, -p_hor, 1, "upper"]
    lower_x_pad = [-p_vert, x_pad_NCHW[2], 1, p_hor, -p_hor, 1, "lower"]
    left_x_pad = [p_vert, -p_vert, 1, 0, p_hor, 1, "left"]
    right_x_pad = [p_vert, -p_vert, 1, -p_hor, x_pad_NCHW[3], 1, "right"]
    all_x = [0, x_NCHW[2], 1, 0, x_NCHW[3], 1]
    upper_x = [0, p_vert, 1, 0, x_NCHW[3], 1]
    upper_x_first = [0, 1, 1, 0, x_NCHW[3], 1]
    upper_x_rev = [0, p_vert, -1, 0, x_NCHW[3], 1]
    upper_x_refl = [1, p_vert + 1, -1, 0, x_NCHW[3], 1]
    lower_x = [-p_vert, x_NCHW[2], 1, 0, x_NCHW[3], 1]
    lower_x_last = [-1, x_NCHW[2], 1, 0, x_NCHW[3], 1]
    lower_x_rev = [-p_vert, x_NCHW[2], -1, 0, x_NCHW[3], 1]
    lower_x_refl = [-p_vert - 1, x_NCHW[2] - 1, -1, 0, x_NCHW[3], 1]
    left_x = [0, x_NCHW[2], 1, 0, p_hor, 1]
    left_x_first = [0, x_NCHW[2], 1, 0, 1, 1]
    left_x_rev = [0, x_NCHW[2], 1, 0, p_hor, -1]
    left_x_refl = [0, x_NCHW[2], 1, 1, p_hor + 1, -1]
    right_x = [0, x_NCHW[2], 1, -p_hor, x_NCHW[3], 1]
    right_x_last = [0, x_NCHW[2], 1, -1, x_NCHW[3], 1]
    right_x_rev = [0, x_NCHW[2], 1, -p_hor, x_NCHW[3], -1]
    right_x_refl = [0, x_NCHW[2], 1, -p_hor - 1, x_NCHW[3] - 1, -1]
    zero_pad = [None, None, None, None]
    pad_tests = [
        {
            "circular": [center_x_pad, all_x],
            "constant": [center_x_pad, all_x],
            "symmetric": [center_x_pad, all_x],
            "reflect": [center_x_pad, all_x],
            "replicate": [center_x_pad, all_x],
        },
        {
            "circular": [upper_x_pad, lower_x],
            "constant": [upper_x_pad, zero_pad],
            "symmetric": [upper_x_pad, upper_x_rev],
            "reflect": [upper_x_pad, upper_x_refl],
            "replicate": [upper_x_pad, upper_x_first],
        },
        {
            "circular": [lower_x_pad, upper_x],
            "constant": [lower_x_pad, zero_pad],
            "symmetric": [lower_x_pad, lower_x_rev],
            "reflect": [lower_x_pad, lower_x_refl],
            "replicate": [lower_x_pad, lower_x_last],
        },
        {
            "circular": [left_x_pad, right_x],
            "constant": [left_x_pad, zero_pad],
            "symmetric": [left_x_pad, left_x_rev],
            "reflect": [left_x_pad, left_x_refl],
            "replicate": [left_x_pad, left_x_first],
        },
        {
            "circular": [right_x_pad, left_x],
            "constant": [right_x_pad, zero_pad],
            "symmetric": [right_x_pad, right_x_rev],
            "reflect": [right_x_pad, right_x_refl],
            "replicate": [right_x_pad, right_x_last],
        },
    ]

    for test_pad in pad_tests:
        compare(
            x_pad,
            x,
            index_x=test_pad[padding_tested][0],
            index_x_ref=test_pad[padding_tested][1],
        )


@pytest.mark.skipif(
    hasattr(PadConv2d, "unavailable_class"),
    reason="PadConv2d not available",
)
@pytest.mark.parametrize(
    "padding_tested",
    ["circular", "constant", "symmetric", "reflect", "replicate", "same", "valid"],
)
@pytest.mark.parametrize(
    "input_shape, batch_size, kernel_size, filters",
    [
        ((1, 5, 5), 250, (3, 3), 2),
        ((5, 5, 5), 250, (3, 3), 2),
        ((2, 7, 7), 250, (7, 7), 2),
        ((128, 5, 5), 250, 3, 2),
    ],
)
def test_predict(padding_tested, input_shape, batch_size, kernel_size, filters):
    """Compare predictions between pad+Conv2d and PadConv2d layers."""
    in_ch = input_shape[0]
    input_shape = uft.to_framework_channel(input_shape)

    if not uft.is_supported_padding(padding_tested,PadConv2d):
        pytest.skip(f"Padding {padding_tested} not supported")
    layer_params = {
        "out_channels": 2,
        "in_channels": in_ch,
        "kernel_size": (3, 3),
        "bias": False,
        "padding": 0,
        "padding_mode": "zeros",
    }
    layer_params["kernel_size"] = kernel_size
    layer_params["out_channels"] = filters
    if isinstance(kernel_size, (int, float)):
        ks = kernel_size
    else:
        ks = kernel_size[0]
    x = np.random.normal(size=(batch_size,) + input_shape).astype("float32")
    x = uft.to_tensor(x)
    x_pad = pad_input(x, padding_tested, layer_params["kernel_size"])
    layer_params_ref = layer_params.copy()
    if padding_tested.lower() == "same":
        layer_params_ref["padding"] = ks // 2  # same

    model_ref = uft.generate_k_lip_model(
        layer_type=tConv2d,
        layer_params=layer_params_ref,
        input_shape=x_pad.shape[1:],
        k=1.0,
    )
    y_ref = uft.compute_predict(model_ref, x_pad)

    layer_params_pad = layer_params.copy()

    if padding_tested.lower() == "valid":
        layer_params_pad["padding"] = 0
    else:
        layer_params_pad["padding"] = ks // 2
    layer_params_pad["padding_mode"] = padding_tested
    model = uft.generate_k_lip_model(
        layer_type=PadConv2d,
        layer_params=layer_params_pad,
        input_shape=input_shape,
        k=1.0,
    )
    uft.copy_model_parameters(model_ref, model)
    y = uft.compute_predict(model, x)
    y_ref = uft.to_numpy(y_ref)
    y = uft.to_numpy(y)

    np.testing.assert_allclose(y_ref, y, 1e-2, 0)


@pytest.mark.skipif(
    hasattr(PadConv2d, "unavailable_class"),
    reason="PadConv2d not available",
)
@pytest.mark.parametrize(
    "padding_tested",
    ["circular", "constant", "symmetric", "reflect", "replicate", "same", "valid"],
)
@pytest.mark.parametrize(
    "input_shape, batch_size, kernel_size, filters",
    [
        ((1, 5, 5), 250, (3, 3), 2),
        ((5, 5, 5), 250, (3, 3), 2),
        ((2, 7, 7), 250, (7, 7), 2),
        ((128, 5, 5), 250, 3, 2),
    ],
)
def test_vanilla(padding_tested, input_shape, batch_size, kernel_size, filters):
    """Compare predictions between PadConv2d and its vanilla export."""
    in_ch = input_shape[0]
    input_shape = uft.to_framework_channel(input_shape)

    if not uft.is_supported_padding(padding_tested,PadConv2d):
        pytest.skip(f"Padding {padding_tested} not supported")
    layer_params = {
        "out_channels": 2,
        "in_channels": in_ch,
        "kernel_size": (3, 3),
        "bias": False,
        "padding": 0,
        "padding_mode": "zeros",
    }
    layer_params["kernel_size"] = kernel_size
    layer_params["out_channels"] = filters
    if isinstance(kernel_size, (int, float)):
        ks = kernel_size
    else:
        ks = kernel_size[0]
    x = np.random.normal(size=(batch_size,) + input_shape).astype("float32")
    x = uft.to_tensor(x)
    layer_params_pad = layer_params.copy()
    if padding_tested.lower() == "valid":
        layer_params_pad["padding"] = 0
    else:
        layer_params_pad["padding"] = ks // 2
    layer_params_pad["padding_mode"] = padding_tested
    model = uft.generate_k_lip_model(
        layer_type=PadConv2d,
        layer_params=layer_params_pad,
        input_shape=input_shape,
        k=1.0,
    )
    y = uft.compute_predict(model, x)

    if uft.vanilla_require_a_copy():
        model2 = uft.generate_k_lip_model(
            layer_type=PadConv2d,
            layer_params=layer_params_pad,
            input_shape=input_shape,
            k=1.0,
        )

        uft.copy_model_parameters(model, model2)
        model_v = vanillaModel(model2)
    else:
        model_v = vanillaModel(model)
    y_v = uft.compute_predict(model_v, x)
    y_v = uft.to_numpy(y_v)
    y = uft.to_numpy(y)
    np.testing.assert_allclose(y_v, y, 1e-2, 0)
