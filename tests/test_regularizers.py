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

from unittest import TestCase

import keras.ops as K
import numpy as np

from deel.lip.regularizers import Lorth2D, LorthRegularizer


class TestLorth2D(TestCase):
    """Tests on Lorth2D class"""

    def test_set_kernel_shape(self):
        """Assert that set_kernel_shape() correctly sets parameters."""
        # RO case
        lorth = Lorth2D()
        self.assertIsNone(lorth.kernel_shape)
        self.assertIsNone(lorth.delta)
        self.assertIsNone(lorth.padding)

        kernel_shape = (5, 5, 32, 64)
        lorth.set_kernel_shape(kernel_shape)
        self.assertEqual(lorth.kernel_shape, kernel_shape)
        self.assertEqual(lorth.delta, 32)
        self.assertEqual(lorth.padding, 4)

        # CO case
        lorth = Lorth2D(stride=3)
        kernel_shape = (5, 5, 8, 70)
        lorth.set_kernel_shape(kernel_shape)

        self.assertEqual(lorth.kernel_shape, kernel_shape)
        self.assertEqual(lorth.delta, 0)  # zero, because CO case
        self.assertEqual(lorth.padding, 3)

    def test_existence_orthogonal_conv(self):
        """Assert that Error and Warning are correctly raised when checking existence
        of the orthogonal convolution."""

        # RO case
        with self.assertRaisesRegex(RuntimeError, "Impossible RO"):
            kernel_shape = (3, 3, 32, 289)
            Lorth2D(kernel_shape, stride=4)

        # CO case
        with self.assertRaisesRegex(RuntimeError, "Impossible CO"):
            kernel_shape = (3, 3, 8, 129)
            Lorth2D(kernel_shape, stride=4)

        # square case (RO=CO)
        with self.assertWarnsRegex(Warning, "hard to optimize"):
            kernel_shape = (3, 3, 32, 32)
            Lorth2D(kernel_shape, stride=1)

    def test_compute_lorth(self):
        """Assert Lorth2D computation on an identity kernel => must return 0"""

        def _identity_kernel(kernel_size, channels):
            assert kernel_size & 1
            kernel_shape = (kernel_size, kernel_size, channels, channels)
            w = np.zeros(kernel_shape, dtype=np.float32)
            for i in range(channels):
                w[kernel_size // 2, kernel_size // 2, i, i] = 1
            return K.convert_to_tensor(w)

        kernel_size = (np.random.randint(1, 10) * 2) + 1
        channels = np.random.randint(1, 100)
        w = _identity_kernel(kernel_size, channels)

        lorth = Lorth2D(w.shape)
        loss = lorth.compute_lorth(w)
        self.assertAlmostEqual(loss.numpy(), 0)


class TestLorthRegularizer(TestCase):
    """Tests on LorthRegularizer"""

    def test_init(self):
        """Check initialization from scratch and from config."""

        kshape1 = (3, 3, 16, 32)
        kshape2 = (5, 5, 16, 8)
        stride = np.random.randint(1, 4)
        lambda_lorth = np.random.normal()
        regul = LorthRegularizer(kshape1, stride, lambda_lorth)

        self.assertEqual(regul.lorth.kernel_shape, kshape1)

        # Set new shape and get config
        regul.set_kernel_shape(kshape2)
        config = regul.get_config()
        new_regul = LorthRegularizer.from_config(config)

        self.assertEqual(new_regul.lorth.kernel_shape, kshape2)
        self.assertEqual(new_regul.stride, stride)
        self.assertEqual(new_regul.lambda_lorth, lambda_lorth)

    def test_1D_not_supported(self):
        """Assert that 1D convolutions are not supported."""
        with self.assertRaisesRegex(NotImplementedError, "Only 2D convolutions"):
            LorthRegularizer(dim=1)
