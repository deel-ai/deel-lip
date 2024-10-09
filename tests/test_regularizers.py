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
from tests.utils_framework import Lorth2d, LorthRegularizer


@pytest.mark.skipif(
    hasattr(Lorth2d, "unavailable_class"),
    reason="Lorth2d not available",
)
@pytest.mark.parametrize(
    "kernel_shape,stride,delta,padding",
    [
        (None, None, None, None),
        ((5, 5, 32, 64), None, 32, 4),  # RO case
        ((5, 5, 8, 70), 3, 0, 3),  # CO case
    ],
)
def test_set_kernel_shape(kernel_shape, stride, delta, padding):
    """Assert that set_kernel_shape() correctly sets parameters."""
    if stride is None:
        lorth = uft.get_instance_framework(Lorth2d, {})
    else:
        lorth = uft.get_instance_framework(Lorth2d, {"stride": stride})
    if kernel_shape is not None:
        lorth.set_kernel_shape(kernel_shape)
    assert lorth.kernel_shape == kernel_shape
    assert lorth.delta == delta
    assert lorth.padding == padding

    # self.assertIsNone(lorth.kernel_shape)
    # self.assertIsNone(lorth.delta)
    # self.assertIsNone(lorth.padding)

    # kernel_shape = (5, 5, 32, 64)
    # lorth.set_kernel_shape(kernel_shape)
    # self.assertEqual(lorth.kernel_shape, kernel_shape)
    # self.assertEqual(lorth.delta, 32)
    # self.assertEqual(lorth.padding, 4)

    # # CO case
    # lorth = Lorth2d(stride=3)
    # kernel_shape = (5, 5, 8, 70)
    # lorth.set_kernel_shape(kernel_shape)

    # self.assertEqual(lorth.kernel_shape, kernel_shape)
    # self.assertEqual(lorth.delta, 0)  # zero, because CO case
    # self.assertEqual(lorth.padding, 3)


@pytest.mark.skipif(
    hasattr(Lorth2d, "unavailable_class"),
    reason="Lorth2d not available",
)
@pytest.mark.parametrize(
    "kernel_shape,stride,err,err_msg",
    [
        ((3, 3, 32, 289), 4, RuntimeError, "Impossible RO"),
        ((3, 3, 8, 129), 4, RuntimeError, "Impossible CO"),
        ((3, 3, 32, 32), 1, Warning, "hard to optimize"),
        ((4, 4, 32, 32), 1, AssertionError, "odd kernel only"),
    ],
)
def test_existence_orthogonal_conv(kernel_shape, stride, err, err_msg):
    """Assert that Error and Warning are correctly raised when checking existence
    of the orthogonal convolution."""

    # RO case
    if err != Warning:
        with pytest.raises(err):
            uft.get_instance_framework(
                Lorth2d, {"kernel_shape": kernel_shape, "stride": stride}
            )
    else:
        with pytest.warns(err):
            uft.get_instance_framework(
                Lorth2d, {"kernel_shape": kernel_shape, "stride": stride}
            )
        # # CO case
        # with self.assertRaisesRegex(RuntimeError, "Impossible CO"):
        #     kernel_shape = (3, 3, 8, 129)
        #     Lorth2d(kernel_shape, stride=4)

        # # square case (RO=CO)
        # with self.assertWarnsRegex(Warning, "hard to optimize"):
        #     kernel_shape = (3, 3, 32, 32)
        #     Lorth2d(kernel_shape, stride=1)


@pytest.mark.skipif(
    hasattr(Lorth2d, "unavailable_class"),
    reason="Lorth2d not available",
)
def test_compute_lorth():
    """Assert Lorth2d computation on an identity kernel => must return 0"""

    def _identity_kernel(kernel_size, channels):
        assert kernel_size & 1
        kernel_shape = (kernel_size, kernel_size, channels, channels)
        w = np.zeros(kernel_shape, dtype=np.float32)
        for i in range(channels):
            w[kernel_size // 2, kernel_size // 2, i, i] = 1
        return w  # tf.constant(w)

    kernel_size = (np.random.randint(1, 10) * 2) + 1
    channels = np.random.randint(1, 100)
    w = _identity_kernel(kernel_size, channels)

    w = uft.to_tensor(w)
    lorth = uft.get_instance_framework(Lorth2d, {"kernel_shape": w.shape})
    loss = lorth.compute_lorth(w)
    np.testing.assert_allclose(loss.numpy(), 0, atol=1e-6)
    # self.assertAlmostEqual(loss.numpy(), 0)


"""
class TestLorthRegularizer(TestCase):
"""
"""Tests on LorthRegularizer"""


@pytest.mark.skipif(
    hasattr(LorthRegularizer, "unavailable_class"),
    reason="LorthRegularizer not available",
)
def test_LorthRegularizer_init():
    """Check initialization from scratch and from config."""

    kshape1 = (3, 3, 16, 32)
    kshape2 = (5, 5, 16, 8)
    stride = np.random.randint(1, 4)
    lambda_lorth = np.random.normal()
    regul = uft.get_instance_framework(
        LorthRegularizer,
        {"kernel_shape": kshape1, "stride": stride, "lambda_lorth": lambda_lorth},
    )
    # LorthRegularizer(kshape1, stride, lambda_lorth)

    assert regul.lorth.kernel_shape == kshape1
    # self.assertEqual(regul.lorth.kernel_shape, kshape1)

    # Set new shape and get config
    regul.set_kernel_shape(kshape2)
    config = regul.get_config()
    new_regul = LorthRegularizer.from_config(config)

    assert new_regul.lorth.kernel_shape == kshape2
    assert new_regul.stride == stride
    assert new_regul.lambda_lorth == lambda_lorth


@pytest.mark.skipif(
    hasattr(LorthRegularizer, "unavailable_class"),
    reason="LorthRegularizer not available",
)
def test_1D_not_supported():
    """Assert that 1D convolutions are not supported."""
    with pytest.raises(NotImplementedError):
        uft.get_instance_framework(LorthRegularizer, {"dim": 1})
