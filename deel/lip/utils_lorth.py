# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Contains utility functions.
"""

from timeit import default_timer as timer
from datetime import timedelta

import tensorflow as tf

from .regularizers import Lorth2D


DEFAULT_NITER_LORTHGRAD = 5
DEFAULT_NITER_CONSTRAINT_LORTHGRAD = 3
DEFAULT_NITER_INITIALIZER_LORTHGRAD = 50
DEFAULT_LAMBDA_LORTHGRAD = 0.2


class Lorth2Dgrad(Lorth2D):
    def __init__(
        self,
        kernel_shape=None,
        stride=1,
        conv_transpose=False,
        niter_newton=DEFAULT_NITER_LORTHGRAD,
        lambda_step=DEFAULT_LAMBDA_LORTHGRAD,
    ) -> None:
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
        self.batch_identity_padded = None
        super(Lorth2Dgrad, self).__init__(kernel_shape, stride, conv_transpose)

        assert not conv_transpose, "Conv transpose not supported for Lorth2Dgrad"
        self.lorthGradient_func = self.lorthGradient_conv
        if self.stride > 1:
            self.lorthGradient_func = self.lorthGradient_depth
        self.niter_newton = niter_newton
        self.lambda_step = lambda_step

    def set_kernel_shape(self, shape):
        super(Lorth2Dgrad, self).set_kernel_shape(shape)
        if (shape is not None) and (self.stride > 1):
            self.batch_identity_padded = self.compute_batch_identity_padded_depth(
                verbose=False
            )

    @tf.function
    def _lorth_forward(self, w):
        convKxK = self._compute_conv_kk(w)
        target = self._compute_target(w, convKxK.shape)

        convKxK_mId = convKxK - target
        lorth_v = tf.reduce_sum(tf.square(convKxK_mId)) - self.delta
        return lorth_v, convKxK_mId

    @tf.function
    def compute_batch_identity_padded_depth(self, verbose=False):
        x_k0, x_k, c_in, c_out = self.kernel_shape
        # stride = 1
        # padding = x_k-1
        batch_identity = tf.eye(x_k * x_k, batch_shape=[c_in])  # c_in, x_k^2, x_k^2
        if verbose:
            print(batch_identity.shape)
        batch_identity = tf.reshape(
            batch_identity, (c_in, x_k, x_k, x_k, x_k)
        )  # c_in, x_k1, x_k2, x_k3, x_k4
        batch_identity = tf.transpose(
            batch_identity, perm=[1, 2, 3, 4, 0], name="transp_id1"
        )  # x_k1, x_k2, x_k3, x_k4, c_in,
        if verbose:
            print(batch_identity.shape)
        # A VERIFIER POUR STRIDE batch_identity_padded =
        # tf.pad(batch_identity,paddings=[[0,0],[0,0],[padding,padding],[padding,padding],[0,0]])
        # #x_k1, x_k2, x_k3+pad, x_k4+pd, c_in,
        batch_identity_padded = tf.pad(
            batch_identity,
            paddings=[
                [0, 0],
                [0, 0],
                [self.padding, self.padding],
                [self.padding, self.padding],
                [0, 0],
            ],
        )  # x_k1, x_k2, x_k3+pad, x_k4+pd, c_in,
        if verbose:
            print(batch_identity_padded.shape)
        if verbose:
            tf.print(
                tf.reduce_max(tf.reduce_sum(batch_identity_padded, axis=[2, 3, 4])),
                tf.reduce_sum(batch_identity_padded, axis=[2, 3, 4]).shape,
            )
        batch_identity_padded = tf.reshape(
            batch_identity_padded, (-1,) + tuple(batch_identity_padded.shape[2:])
        )  # x_k1*x_k2, x_k3+pad, x_k4+pad, c_in
        if verbose:
            print(batch_identity_padded.shape)
            tf.print(
                tf.reduce_min(batch_identity_padded),
                tf.reduce_max(batch_identity_padded),
                tf.reduce_sum(batch_identity_padded),
            )
            if batch_identity_padded.shape[0] == 27:
                tf.print(batch_identity_padded[0:1, :, :, 0], summarize=-1)
                tf.print(batch_identity_padded[0:1, :, :, 1], summarize=-1)
                tf.print(batch_identity_padded[0:1, :, :, 2], summarize=-1)
                tf.print(batch_identity_padded[1:2, :, :, 0], summarize=-1)
                tf.print(batch_identity_padded[1:2, :, :, 1], summarize=-1)
                tf.print(batch_identity_padded[1:2, :, :, 2], summarize=-1)
                tf.print(batch_identity_padded[-1:, :, :, 0], summarize=-1)
                tf.print(batch_identity_padded[-1:, :, :, 1], summarize=-1)
                tf.print(batch_identity_padded[-1:, :, :, 2], summarize=-1)
        return batch_identity_padded

    # Gradient computation for stride convolution
    # (using depthwise convolution due to independence)
    @tf.function
    def lorthGradient_depth(
        self, kernel, verbose=False
    ):  # , stride = 1, delta = 0.0, verbose = False):
        if verbose:
            start = timer()

        x_k0, x_k, c_in, c_out = self.kernel_shape

        # padding = x_k-1
        lorth_v, convKxK_mId = self._lorth_forward(
            kernel
        )  # c_out,x_k+pad, x_k+pad, c_out

        bip_depth = (
            self.batch_identity_padded
        )  # compute_batch_identity_padded_test_depthconv(kernel.shape, verbose = True)

        if verbose:
            print("convKxK_mId", convKxK_mId.shape)
            print("batch_identity_padded", self.batch_identity_padded.shape)

        convKxId_depth = tf.nn.depthwise_conv2d(
            bip_depth,
            kernel,
            strides=[1, self.stride, self.stride, 1],
            padding="VALID",
            name="convKxId_depth",
        )  # x_k*x_k, x_k+pad, x_k+pad, c_in*c_out
        if verbose:
            print("convKxId_depth", convKxId_depth.shape)

        convKxId_depth_r = tf.reshape(
            convKxId_depth, convKxId_depth.shape[0:3] + (c_in, c_out)
        )  # x_k*x_k, x_k+pad, x_k+pad, c_in, c_out
        convKxId_depth_r = tf.transpose(
            convKxId_depth_r, perm=[1, 2, 4, 3, 0]
        )  # x_k+pad, x_k+pad, c_out, c_in, x_k*x_k,
        convKxId_depth_r2 = tf.reshape(
            convKxId_depth_r, (-1,) + tuple(convKxId_depth_r.shape[3:])
        )  # (x_k+pad)*(x_k+pad)*c_out , c_in, x_k*x_k,

        convKxK_mId_r = tf.reshape(
            convKxK_mId,
            (
                c_out,
                -1,
            ),
        )  # c_out,(x_k+padding)*(x_k+padding)*c_out

        grad_part1_depth = 2 * tf.tensordot(
            convKxK_mId_r, convKxId_depth_r2, axes=1, name="td_jpart1_depth"
        )  # <dconv(w_i,pad_w_j)/dw_j,Conv(K,K)-Id> # c_out, c_in, x_k*x_k

        convKxId_depth_flipped = tf.reverse(
            convKxId_depth_r, axis=[0, 1]
        )  # R(x_k+pad), R(x_k+pad), c_out, c_in, x_k*x_k
        if verbose:
            print("grad_part1_depth ", grad_part1_depth.shape)

        convKxId_depth_flipped_t2 = tf.reshape(
            convKxId_depth_flipped, (-1, c_in, x_k * x_k)
        )  # R(x_k+pad)*R(x_k+pad)*c_out, c_in, x_k*x_k

        convKxK_mId_T_depth = tf.transpose(
            convKxK_mId, perm=[3, 1, 2, 0]
        )  # c_out-2,(x_k+pad),(x_k+pad),c_out-1
        convKxK_mId_T_depth = tf.reshape(
            convKxK_mId_T_depth, (c_out, -1)
        )  # c_out-2, (x_k+pad)*(x_k+pad)*c_out-1

        # grad_part2 = 2*tf.tensordot(convKxId_flipped_t2,convKxK_mId_T,
        # axes=1,name="td_jpart2")
        # <dconv(w_i,pad_w_j)/dw_i,(Conv(K,K)-Id)^T>
        # c_in, x_k, x_k, c_out-2
        grad_part2_depth = 2 * tf.tensordot(
            convKxK_mId_T_depth,
            convKxId_depth_flipped_t2,
            axes=1,
            name="td_jpart2_depth",
        )  # <dconv(w_i,pad_w_j)/dw_i,(Conv(K,K)-Id)^T>  #c_out-2, c_in, x_k*x_k,

        if verbose:
            print("grad_part2_depth ", grad_part2_depth.shape)

        jacobian_depth_t3 = (
            grad_part1_depth + grad_part2_depth
        )  # Sum for dconv(w_i,pad_w_i)/dw_i #c_out-2, c_in, x_k*x_k,
        jacobian_depth = tf.transpose(
            jacobian_depth_t3, perm=[2, 1, 0], name="final_transp"
        )  # x_k*x_k, c_in, c_out-2
        jacobian_depth = tf.reshape(
            jacobian_depth, (x_k, x_k, c_in, c_out)
        )  # x_k, x_k, c_in, c_out-2

        # print(tf.norm(jacobian-jacobian_depth))
        if verbose:
            print("jacobian_depth ", jacobian_depth.shape)

        if verbose:
            end = timer()
            print(timedelta(seconds=end - start))

        return jacobian_depth, lorth_v

    @tf.function
    def lorthGradient_conv(
        self, kernel, verbose=False
    ):  # , stride = 1, delta = 0.0, verbose = False):
        if verbose:
            start = timer()

        x_k0, x_k, c_in, c_out = self.kernel_shape
        # padding = x_k-1
        lorth_v, convKxK_mId = self._lorth_forward(kernel)
        # c_out,x_k+pad, x_k+pad, c_out

        kernel_t = tf.transpose(kernel, perm=[0, 1, 3, 2])  # x_k, x_k, c_out, c_in
        kernel_tr = tf.reverse(kernel_t, axis=[0, 1])  # x_k, x_k, c_out, c_in
        grad_part1_conv2d = 2 * tf.nn.conv2d(
            convKxK_mId,
            kernel_tr,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="convKxK_KT",
        )  # c_in*x_k*x_k, x_k+pad, x_k+pad, c_out

        convKxK_mId_T_depth1 = tf.transpose(
            convKxK_mId, perm=[3, 1, 2, 0]
        )  # c_out-2,(x_k+pad),(x_k+pad),c_out-1

        grad_part2_conv2d = 2 * tf.nn.conv2d(
            convKxK_mId_T_depth1,
            kernel_t,
            strides=[1, self.stride, self.stride, 1],
            padding="VALID",
            name="convKxK_KT",
        )  # c_out-2,  x_k, x_k, c_in,
        grad_part2_conv2d = tf.reverse(grad_part2_conv2d, axis=[1, 2])

        jacobian_conv_t3 = (
            grad_part1_conv2d + grad_part2_conv2d
        )  # Sum for dconv(w_i,pad_w_i)/dw_i #c_out-2,x_k,x_k,c_in
        jacobian_conv = tf.transpose(
            jacobian_conv_t3, perm=[1, 2, 3, 0], name="final_transp"
        )  # x_k,x_k, c_in, c_out-2

        if verbose:
            print("jacobian_conv ", jacobian_conv.shape)

        if verbose:
            end = timer()
            print(timedelta(seconds=end - start))

        return jacobian_conv, lorth_v

    def lorthGradient_orthogonalization(self, kernel, verbose=False):
        """stride = 1,
        delta = 0.0,"""
        W_bar = kernel
        # print(niter_bjorck,kernel.shape,delta)
        # tf.print("start iterations ",niter_bjorck)
        for it in range(self.niter_newton):
            # grad_lorth_ref = lorthGradient_ReferenceMatrixMultiply(W_bar)
            grad_lorth, lorth_v = self.lorthGradient_func(W_bar)

            # adaptative step: function of lorth_v, and norm_W/norm_grad
            factor = tf.norm(W_bar) / tf.norm(
                grad_lorth
            )  # In case tf.norm(grad_lorth) is high
            factor = tf.minimum(lorth_v, 0.1) * factor  # 1.0
            lambda_f = tf.stop_gradient(tf.minimum(self.lambda_step, factor))
            if verbose:
                tf.print(
                    it,
                    ":",
                    lorth_v,
                    factor,
                    lambda_f,
                    tf.norm(W_bar),
                    tf.norm(grad_lorth),
                    W_bar.shape,
                )

            W_bar = W_bar - lambda_f * grad_lorth
        return W_bar
