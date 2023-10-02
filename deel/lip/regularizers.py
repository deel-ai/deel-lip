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
"""
This module contains custom Keras regularizers. They can be used as kernel regularizer
in any Keras layer.
"""
import warnings
from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.utils import register_keras_serializable


class Lorth(ABC):
    def __init__(self, dim, kernel_shape=None, stride=1, conv_transpose=False) -> None:
        """
        Base class for Lorth regularization. Not meant to be used standalone.

        Ref. Achour & al., Existence, Stability And Scalability Of Orthogonal
        Convolutional Neural Networks (2022).
        https://www.jmlr.org/papers/v23/22-0026.html

        Args:
            dim (int): the rank of the convolution, e.g. "2" for 2D convolution.
            kernel_shape: the shape of the kernel.
            stride (int): stride used in the associated convolution
            conv_transpose (bool): whether the kernel is from a transposed convolution.
        """
        super(Lorth, self).__init__()
        self.dim = dim
        self.stride = stride
        self.conv_transpose = conv_transpose
        self.set_kernel_shape(kernel_shape)

    def _get_kernel_shape(self):
        """Return the kernel size, the number of input channels and output channels"""
        return [self.kernel_shape[i] for i in (0, -2, -1)]

    def _compute_delta(self):
        """delta is positive in CO case, zero in RO case."""
        _, C, M = self._get_kernel_shape()
        if not self.conv_transpose:
            delta = M - (self.stride**self.dim) * C
        else:
            delta = C - (self.stride**self.dim) * M
        return max(0, delta)

    def _alphaNormSpectral(self):
        """delta is positive in CO case, zero in RO case."""
        R, C1, M1 = self._get_kernel_shape()
        if not self.conv_transpose:
            C, M = C1, M1
        else:
            C, M = M1, C1
        alpha = (
            R * C * M
        )  # huge value to get the minimum value in case of square matrix
        if M - (self.stride**self.dim) * C <= 0:  # RO case
            alpha = M * (2 * ((R - 1) // self.stride) + 1) ** self.dim
        if M - (self.stride**self.dim) * C >= 0:  # CO case
            alpha = min(alpha, C * (2 * R - 1) ** self.dim)
        return alpha

    def _check_if_orthconv_exists(self):
        """check the existence of Orthogonal convolution (for circular padding)"""
        R, C, M = self._get_kernel_shape()
        msg = "Impossible {} configuration for orthogonal convolution."
        if C * self.stride**self.dim >= M:  # RO case
            if M > C * (R**self.dim):
                raise RuntimeError(msg.format("RO"))
        else:  # CO case
            if self.stride > R:
                raise RuntimeError(msg.format("CO"))
        if C * (self.stride**self.dim) == M:  # square case
            warnings.warn(
                "LorthRegularizer: warning configuration C*S^2=M is hard to optimize."
            )

    def set_kernel_shape(self, shape):
        if shape is None:
            self.kernel_shape, self.padding, self.delta = None, None, None
            return

        R = shape[0]
        self.kernel_shape = shape
        self.padding = ((R - 1) // self.stride) * self.stride
        self.delta = self._compute_delta()
        self.alphaNormSpectral = self._alphaNormSpectral()

        # Assertions on kernel shape and existence of orthogonal convolution
        assert R & 1, "Lorth regularizer requires odd kernels. Receives " + str(R)
        self._check_if_orthconv_exists()

    @abstractmethod
    def _compute_conv_kk(self, w):
        raise NotImplementedError()

    @abstractmethod
    def _compute_target(self, w, output_shape):
        raise NotImplementedError()

    def compute_lorth(self, w):
        output = self._compute_conv_kk(w)
        target = self._compute_target(w, output.shape)
        return tf.reduce_sum(tf.square(output - target)) - self.delta


class Lorth2D(Lorth):
    def __init__(self, kernel_shape=None, stride=1, conv_transpose=False) -> None:
        """
        Lorth computation for 2D convolutions. Although this class allows to compute
        the regularization term, it cannot be used as it is in a layer.

        Ref. Wang & al., Orthogonal Convolutional Neural Networks (2020).
        http://arxiv.org/abs/1911.12207

        Args:
            kernel_shape: the shape of the kernel.
            stride (int): stride used in the associated convolution
            conv_transpose (bool): whether the kernel is from a transposed convolution.
        """
        dim = 2
        super(Lorth2D, self).__init__(dim, kernel_shape, stride, conv_transpose)

    def _compute_conv_kk(self, w):
        w_reshape = tf.transpose(w, perm=[3, 0, 1, 2])
        w_padded = tf.pad(
            w_reshape,
            paddings=[
                [0, 0],
                [self.padding, self.padding],
                [self.padding, self.padding],
                [0, 0],
            ],
        )
        return tf.nn.conv2d(w_padded, w, self.stride, padding="VALID")

    def _compute_target(self, w, convKxK_shape):
        C_out = w.shape[-1]
        outm3 = convKxK_shape[-3]
        outm2 = convKxK_shape[-2]
        ct = tf.cast(tf.math.floor(outm2 / 2), dtype=tf.int32)

        target_zeros = tf.zeros((outm3 * outm2 - 1, C_out, C_out))
        target = tf.concat(
            [
                target_zeros[: ct * outm2 + ct],
                tf.expand_dims(tf.eye(C_out), axis=0),
                target_zeros[ct * outm2 + ct :],
            ],
            axis=0,
        )

        target = tf.reshape(target, (outm3, outm2, C_out, C_out))
        target = tf.transpose(target, [2, 0, 1, 3])
        return target


@register_keras_serializable("deel-lip", "LorthRegularizer")
class LorthRegularizer(Regularizer):
    def __init__(
        self,
        kernel_shape=None,
        stride=1,
        lambda_lorth=1.0,
        dim=2,  # 2 for 2D conv, 1 for 1D conv
        conv_transpose=False,
    ) -> None:
        """
        Regularize a conv kernel to be orthogonal (all singular values are equal to 1)
        using Lorth regularizer.

        Args:
            kernel_shape: the shape of the kernel.
            stride (int): stride used in the associated convolution
            lambda_lorth (float): weight of the orthogonalization regularization.
            dim (int): 1 for 1D convolutions, 2 for 2D convolutions. Defaults to 2.
            conv_transpose (bool): whether the kernel is from a transposed convolution.
        """
        super(LorthRegularizer, self).__init__()
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.lambda_lorth = lambda_lorth
        self.dim = dim
        self.conv_transpose = conv_transpose
        if self.dim == 2:
            self.lorth = Lorth2D(kernel_shape, stride, conv_transpose)
        else:
            raise NotImplementedError("Only 2D convolutions are supported for Lorth.")

    def set_kernel_shape(self, shape):
        self.kernel_shape = shape
        self.lorth.set_kernel_shape(shape)

    def __call__(self, x):
        return self.lambda_lorth * self.lorth.compute_lorth(x)

    def get_config(self):
        return {
            "kernel_shape": self.kernel_shape,
            "stride": self.stride,
            "lambda_lorth": self.lambda_lorth,
            "dim": self.dim,
            "conv_transpose": self.conv_transpose,
        }


@register_keras_serializable("deel-lip", "OrthDenseRegularizer")
class OrthDenseRegularizer(Regularizer):
    def __init__(self, lambda_orth=1.0) -> None:
        """
        Regularize a Dense kernel to be orthogonal (all singular values are equal to 1)
        minimizing W.W^T-Id

        Args:
            lambda_orth (float): regularization factor (must be positive)
        """
        super(OrthDenseRegularizer, self).__init__()
        self.lambda_orth = lambda_orth

    def _dense_orth_dist(self, w):
        transp_b = w.shape[0] <= w.shape[1]
        # W.W^T if h<=w; W^T.W otherwise
        wwt = tf.matmul(w, w, transpose_a=not transp_b, transpose_b=transp_b)
        idx = tf.eye(wwt.shape[0])
        return tf.reduce_sum(tf.square(wwt - idx))

    def __call__(self, x):
        return self.lambda_orth * self._dense_orth_dist(x)

    def get_config(self):
        return {"lambda_orth": self.lambda_orth}
