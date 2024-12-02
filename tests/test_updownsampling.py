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
from .utils_framework import InvertibleDownSampling, InvertibleUpSampling


def check_downsample(x, y, kernel_size):
    index = 0
    for dx in range(kernel_size):
        for dy in range(kernel_size):
            xx = x[:, :, dx::kernel_size, dy::kernel_size]
            yy = y[:, index :: (kernel_size * kernel_size), :, :]
            np.testing.assert_almost_equal(xx, yy, decimal=6)
            index += 1


@pytest.mark.skipif(
    hasattr(InvertibleDownSampling, "unavailable_class"),
    reason="InvertibleDownSampling not available",
)
def test_invertible_downsample():
    x_np = np.arange(32).reshape(1, 2, 4, 4)
    x = uft.to_NCHW_inv(x_np)
    x = uft.to_tensor(x)
    dw_layer = uft.get_instance_framework(InvertibleDownSampling, {"kernel_size": 2})
    y = dw_layer(x)
    y_np = uft.to_numpy(y)
    y_np = uft.to_NCHW(y_np)
    assert y_np.shape == (1, 8, 2, 2)
    check_downsample(x_np, y_np, 2)

    # 2D input
    x_np = np.random.rand(10, 1, 128, 128)  # torch.rand(10, 1, 128, 128)
    x = uft.to_NCHW_inv(x_np)
    x = uft.to_tensor(x)

    dw_layer = uft.get_instance_framework(InvertibleDownSampling, {"kernel_size": 4})
    y = dw_layer(x)
    y_np = uft.to_numpy(y)
    y_np = uft.to_NCHW(y_np)
    assert y_np.shape == (10, 16, 32, 32)
    check_downsample(x_np, y_np, 4)

    x_np = np.random.rand(10, 4, 64, 64)
    x = uft.to_NCHW_inv(x_np)
    x = uft.to_tensor(x)
    dw_layer = uft.get_instance_framework(InvertibleDownSampling, {"kernel_size": 2})
    y = dw_layer(x)
    y_np = uft.to_numpy(y)
    y_np = uft.to_NCHW(y_np)
    assert y_np.shape == (10, 16, 32, 32)
    check_downsample(x_np, y_np, 2)


@pytest.mark.skipif(
    hasattr(InvertibleUpSampling, "unavailable_class"),
    reason="InvertibleUpSampling not available",
)
def test_invertible_upsample():

    # 2D input
    x_np = np.random.rand(10, 16, 32, 32)
    x = uft.to_NCHW_inv(x_np)
    x = uft.to_tensor(x)
    dw_layer = uft.get_instance_framework(InvertibleUpSampling, {"kernel_size": 4})
    y = dw_layer(x)
    y_np = uft.to_numpy(y)
    y_np = uft.to_NCHW(y_np)
    assert y_np.shape == (10, 1, 128, 128)
    check_downsample(y_np, x_np, 4)
    
    dw_layer = uft.get_instance_framework(InvertibleUpSampling, {"kernel_size": 2})
    y = dw_layer(x)
    y_np = uft.to_numpy(y)
    y_np = uft.to_NCHW(y_np)
    assert y_np.shape == (10, 4, 64, 64)
    check_downsample(y_np, x_np, 2)


@pytest.mark.skipif(
    hasattr(InvertibleUpSampling, "unavailable_class")
    or hasattr(InvertibleDownSampling, "unavailable_class"),
    reason="InvertibleUpSampling not available",
)
def test_invertible_upsample_downsample():
    x_np = np.random.rand(10, 16, 32, 32)
    x = uft.to_NCHW_inv(x_np)
    x = uft.to_tensor(x)
    up_layer = uft.get_instance_framework(InvertibleUpSampling, {"kernel_size": 4})
    y = up_layer(x)

    dw_layer = uft.get_instance_framework(InvertibleDownSampling, {"kernel_size": 4})
    z = dw_layer(y)
    assert z.shape == x.shape
    np.testing.assert_array_equal(x, z)

    x_np = np.random.rand(10, 1, 128, 128)  # torch.rand(10, 1, 128, 128)
    x = uft.to_NCHW_inv(x_np)
    x = uft.to_tensor(x)

    dw_layer = uft.get_instance_framework(InvertibleDownSampling, {"kernel_size": 4})
    y = dw_layer(x)
    up_layer = uft.get_instance_framework(InvertibleUpSampling, {"kernel_size": 4})
    z = up_layer(y)
    assert z.shape == x.shape
    np.testing.assert_array_equal(x, z)
