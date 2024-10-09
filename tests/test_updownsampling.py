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
from .utils_framework import invertible_downsample, invertible_upsample


@pytest.mark.skipif(
    hasattr(invertible_downsample, "unavailable_class"),
    reason="invertible_downsample not available",
)
def test_invertible_downsample():
    # 1D input
    x = uft.to_tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]])
    x = uft.get_instance_framework(
        invertible_downsample, {"input": x, "kernel_size": (2,)}
    )
    assert x.shape == (1, 4, 2)

    # TODO: Check this.
    np.testing.assert_equal(uft.to_numpy(x), [[[1, 2], [3, 4], [5, 6], [7, 8]]])

    # 2D input
    x = np.random.rand(10, 1, 128, 128)  # torch.rand(10, 1, 128, 128)
    x = uft.to_tensor(x)
    assert invertible_downsample(x, (4, 4)).shape == (10, 16, 32, 32)

    x = np.random.rand(10, 4, 64, 64)
    x = uft.to_tensor(x)
    assert invertible_downsample(x, (2, 2)).shape == (10, 16, 32, 32)

    # 3D input
    x = np.random.rand(10, 2, 128, 64, 64)
    x = uft.to_tensor(x)
    assert invertible_downsample(x, 2).shape == (10, 16, 64, 32, 32)


@pytest.mark.skipif(
    hasattr(invertible_upsample, "unavailable_class"),
    reason="invertible_upsample not available",
)
def test_invertible_upsample():
    # 1D input
    x = uft.to_tensor([[[1, 2], [3, 4], [5, 6], [7, 8]]])
    x = uft.get_instance_framework(
        invertible_upsample, {"input": x, "kernel_size": (2,)}
    )

    assert x.shape == (1, 2, 4)

    # Check output.
    np.testing.assert_equal(uft.to_numpy(x), [[[1, 2, 3, 4], [5, 6, 7, 8]]])

    # 2D input
    x = np.random.rand(10, 16, 32, 32)
    x = uft.to_tensor(x)
    y = uft.get_instance_framework(
        invertible_upsample, {"input": x, "kernel_size": (4, 4)}
    )
    assert y.shape == (10, 1, 128, 128)
    y = uft.get_instance_framework(
        invertible_upsample, {"input": x, "kernel_size": (2, 2)}
    )
    assert y.shape == (10, 4, 64, 64)

    # 3D input
    x = np.random.rand(10, 16, 64, 32, 32)
    x = uft.to_tensor(x)
    y = uft.get_instance_framework(invertible_upsample, {"input": x, "kernel_size": 2})
    assert y.shape == (10, 2, 128, 64, 64)
