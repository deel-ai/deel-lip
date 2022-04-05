# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import warnings
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.utils import register_keras_serializable


class Lorth(ABC):
    def __init__(
        self,
        kernel_shape=None,
        stride=1,
        flag_deconv=False,
    ) -> None:
        """
        Base class for Lorth regularization. Not meant to be used standalone.

        Args:
            kernel_shape: the shape of the kernel.
            stride: int: stride used in the associated convolution
            flag_deconv: flag used to indicate if the dimension of the output is
                greater or lower than the dimnesion of the input.

        """
        super(Lorth, self).__init__()
        self.stride = stride
        self.kernel_shape = kernel_shape
        self.flag_deconv = flag_deconv
        self.r = 0  # will be set by _set_kernel_shape
        self.delta = 0  # will be set by _set_kernel_shape
        self.dim = None  # will be set by upper class
        # self._set_kernel_shape(self.kernel_shape)  to be done in upper class

    def _get_kernel_shape(self):
        if self.dim == 2:
            (R, R, C, M) = self.kernel_shape
        else:
            (R, C, M) = self.kernel_shape
        return (R, C, M)

    def _compute_delta(self):
        (R, C, M) = self._get_kernel_shape()
        if not self.flag_deconv:
            delta = M - (self.stride**self.dim) * C
        else:
            delta = C - (self.stride**self.dim) * M
        delta = max(0, delta)
        if delta > 0:
            print(
                "Embedding case C=",
                C,
                " M=",
                M,
                " Stride=",
                self.stride,
                " deconv=",
                self.flag_deconv,
                " => Minimum delta: ",
                delta,
            )
        return delta

    def _check_if_orthconv_exists(self):
        (R, C, M) = self._get_kernel_shape()
        # RO case
        if C * self.stride**self.dim >= M:
            if M > C * (R**self.dim):
                raise RuntimeError(
                    "Impossible RO configuration for orthogonal convolution"
                )
        else:
            if self.stride > R:
                raise RuntimeError(
                    "Impossible CO configuration for orthogonal convolution"
                )

        if C * (self.stride**self.dim) == M:
            warnings.warn(
                "LorthRegularizer: Warning configuration C*S^2=M is hard to optimize"
            )

    def _set_kernel_shape(self, shape):
        if shape is None:
            return
        self.kernel_shape = shape
        (R, C, M) = self._get_kernel_shape()
        assert R & 1, "Lorth Regularizer require odd kernels " + str(R)
        self.r = R // 2
        self.padding = (
            (R - 1) // self.stride
        ) * self.stride  # padding size for Lorth convolution
        self.delta = self._compute_delta()
        self._check_if_orthconv_exists()

    @abstractmethod
    def _compute_conv_kk(self, w, verbose=False):
        raise NotImplementedError()

    @abstractmethod
    def _compute_target(self, w, output_shape, verbose=False):
        raise NotImplementedError()

    def _compute_lorth(self, w, verbose=False):
        output = self._compute_conv_kk(w, verbose=verbose)
        target = self._compute_target(w, output.shape, verbose=verbose)
        if verbose:
            print("target ", target.shape)
            print("output ", output.shape)
            # ct = self.compute_ct(output.shape)
            tf.print(tf.reduce_max(tf.abs(output)))
            tf.print(self.delta)
            tf.print(tf.reduce_sum(target))
            tf.print(tf.reduce_sum(tf.square(output - target)) - self.delta)
            # tf.print(output[:,ct,ct,:])
        return tf.reduce_sum(tf.square(output - target)) - self.delta


class Lorth2D(Lorth):
    def __init__(
        self,
        kernel_shape=None,
        stride=1,
        flag_deconv=False,
    ) -> None:
        """
        Lorth computation for 2D convolutions. Although this class allow to compute
        the regularization term, it cannot be used as-is in a layer.

        Args:
            kernel_shape: the shape of the kernel.
            stride: int: stride used in the associated convolution
            flag_deconv: flag used to indicate if the dimension of the output is
                greater or lower than the dimnesion of the input.

        """
        super(Lorth2D, self).__init__(
            kernel_shape=kernel_shape, stride=stride, flag_deconv=flag_deconv
        )
        self.dim = 2
        self._set_kernel_shape(self.kernel_shape)

    def _compute_conv_kk(self, w, verbose=False):
        w_reshape = tf.transpose(w, perm=[3, 0, 1, 2])
        if verbose:
            print("w_reshape ", w_reshape.shape)
        w_padded = tf.pad(
            w_reshape,
            paddings=[
                [0, 0],
                [self.padding, self.padding],
                [self.padding, self.padding],
                [0, 0],
            ],
        )
        if verbose:
            print("w_padded ", w_padded.shape)
        output = tf.nn.conv2d(
            w_padded, w, strides=[1, self.stride, self.stride, 1], padding="VALID"
        )
        if verbose:
            print(output.shape)
        return output

    def compute_ct(self, convKxK_shape):
        outm2 = convKxK_shape[-2]
        ct = int(np.floor(outm2 / 2))
        return ct

    def _compute_target(self, w, convKxK_shape, verbose=False):
        [R, R1, i_c, o_c] = w.shape
        outm3 = convKxK_shape[-3]
        outm2 = convKxK_shape[-2]
        target = tf.zeros((o_c, outm3, outm2, o_c))
        if verbose:
            print("target shape ", target.shape, " with Id (o_c,o_c) ", o_c)
        ct = self.compute_ct(convKxK_shape)

        target_zeros = tf.zeros((outm3 * outm2 - 1, o_c, o_c))
        if verbose:
            print(target_zeros[: ct * outm2 + ct].shape)
            print(tf.expand_dims(tf.eye(o_c), axis=0).shape)
            print(target_zeros[ct * outm2 + ct :].shape)
        target = tf.concat(
            [
                target_zeros[: ct * outm2 + ct],
                tf.expand_dims(tf.eye(o_c), axis=0),
                target_zeros[ct * outm2 + ct :],
            ],
            axis=0,
        )

        if verbose:
            print("target ", target.shape)
        target = tf.reshape(target, (outm3, outm2, o_c, o_c))

        if verbose:
            print("target ", target.shape)
        target = tf.transpose(target, [2, 0, 1, 3])
        if verbose:
            print("target ", target.shape)
        return target


@register_keras_serializable("deel-lip", "LorthRegularizer")
class LorthRegularizer(Regularizer):
    def __init__(
        self,
        kernel_shape=None,
        stride=1,
        lambda_lorth=1.0,
        dim=2,  # 2 for 2D conv, 1 for 1D conv
        flag_deconv=False,
    ) -> None:
        """
        Regularize a conv kernel to be orthogonal (sigma min and max are equal to 1)
        using Lorth regularizer.

        Args:
            kernel_shape: the shape of the kernel.
            stride: int: stride used in the associated convolution
            lambda_lorth: float : weight of the orthogonalization regularization.
            dim: int: default 2: 1 for 1D convolutions, 2 for 2D convolutions.
            flag_deconv: flag used to indicate if the dimension of the output is
                greater or lower than the dimnesion of the input.

        """
        super(LorthRegularizer, self).__init__()
        self.stride = stride
        self.lambda_lorth = lambda_lorth
        self.kernel_shape = kernel_shape
        self.dim = dim
        self.flag_deconv = flag_deconv
        if self.dim == 2:
            self.lorth = Lorth2D(
                kernel_shape=kernel_shape, stride=stride, flag_deconv=flag_deconv
            )
        else:
            raise NotImplementedError

    def _set_kernel_shape(self, shape):
        self.kernel_shape = shape
        self.lorth._set_kernel_shape(shape)

    def __call__(self, x):
        reg = self.lambda_lorth * self.lorth._compute_lorth(x)
        return reg

    def get_config(self):
        return {
            "kernel_shape": self.kernel_shape,
            "stride": self.stride,
            "lambda_lorth": self.lambda_lorth,
            "dim": self.dim,
            "flag_deconv": self.flag_deconv,
        }


@register_keras_serializable("deel-lip", "OrthDenseRegularizer")
class OrthDenseRegularizer(Regularizer):
    def __init__(
        self,
        lambda_orth=1.0,
    ) -> None:
        """
        Regularize a Dense kernel to be orthogonal (sigma min and max =1)
        minimizing W.W^T-Id

        Args:
            lambda_orth: float: weight applied to the regularization (must be
                positive).

        """
        super(OrthDenseRegularizer, self).__init__()
        self.lambda_orth = lambda_orth

    def _dense_orth_dist(self, w):
        transp_b = w.shape[0] <= w.shape[1]
        # print(w.shape)
        wwt = tf.matmul(
            w, w, transpose_a=not transp_b, transpose_b=transp_b
        )  # WW^T if h<=w; W^TW otherwise
        # print(wwt.shape)
        idx = tf.eye(wwt.shape[0])
        return tf.reduce_sum(tf.square(wwt - idx))

    def __call__(self, x):
        reg = self.lambda_orth * self._dense_orth_dist(x)
        return reg

    def get_config(self):
        return {
            "lambda_orth": self.lambda_orth,
        }
