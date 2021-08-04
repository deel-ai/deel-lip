# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains computation function, for Bjorck and spectral
normalization. This is done for internal use only.
"""
import tensorflow as tf
from tensorflow.keras import backend as K

DEFAULT_NITER_BJORCK = 15
DEFAULT_NITER_SPECTRAL = 3
DEFAULT_NITER_SPECTRAL_INIT = 10
DEFAULT_BETA_BJORCK = 0.25


def project_kernel(
    kernel,
    u,
    adjustment_coef,
    niter_spectral=DEFAULT_NITER_SPECTRAL,
    niter_bjorck=DEFAULT_NITER_BJORCK,
    beta=DEFAULT_BETA_BJORCK,
):
    """
    Perform reshaped kernel orthogonalization (RKO) to the kernel given as input. It
    apply the power method to find the largest singular value and apply the Bjorck
    algorithm to the rescaled kernel. This greatly improve the stability and and
    speed convergence of the bjorck algorithm.

    Args:
        kernel: the kernel to orthogonalize
        u: the vector used to do the power iteration method
        adjustment_coef: the adjustment coefficient as used in convolution
        niter_spectral: number of iteration to do in spectral algorithm
        niter_bjorck: iteration used for bjorck algorithm
        beta: the beta used in the bjorck algorithm

    Returns: the orthogonalized kernel, the new u, and sigma which is the largest
        singular value

    """
    W_bar, u, sigma = spectral_normalization(kernel, u, niter=niter_spectral)
    W_bar = bjorck_normalization(W_bar, niter=niter_bjorck, beta=beta)
    W_bar = W_bar * adjustment_coef
    W_bar = K.reshape(W_bar, kernel.shape)
    return W_bar, u, sigma


def bjorck_normalization(w, niter=DEFAULT_NITER_BJORCK, beta=DEFAULT_BETA_BJORCK):
    """
    apply Bjorck normalization on w.

    Args:
        w: weight to normalize, in order to work properly, we must have
            max_eigenval(w) ~= 1
        niter: number of iterations
        beta: beta used in each iteration, must be in the interval ]0, 0.5]

    Returns:
        the orthonormal weights

    """
    for i in range(niter):
        w = (1 + beta) * w - beta * w @ tf.transpose(w) @ w
    return w


def _power_iteration(w, u, niter=DEFAULT_NITER_SPECTRAL):
    """
    Internal function that performs the power iteration algorithm.

    Args:
        w: weights matrix that we want to find eigen vector
        u: initialization of the eigen vector
        niter: number of iteration, must be greater than 0

    Returns:
         u and v corresponding to the maximum eigenvalue

    """
    _u = u
    for i in range(niter):
        _v = tf.math.l2_normalize(_u @ tf.transpose(w))
        _u = tf.math.l2_normalize(_v @ w)
    return _u, _v


def spectral_normalization(kernel, u, niter=DEFAULT_NITER_SPECTRAL):
    """
    Normalize the kernel to have it's max eigenvalue == 1.

    Args:
        kernel: the kernel to normalize
        u: initialization for the max eigen vector
        niter: number of iteration

    Returns:
        the normalized kernel w_bar, it's shape, the maximum eigen vector, and the
        maximum eigen value

    """
    W_shape = kernel.shape
    if u is None:
        niter *= 2  # if u was not known increase number of iterations
        u = tf.ones(shape=tuple([1, W_shape[-1]]))
    # Flatten the Tensor
    W_reshaped = tf.reshape(kernel, [-1, W_shape[-1]])
    _u, _v = _power_iteration(W_reshaped, u, niter)
    # Calculate Sigma
    sigma = _v @ W_reshaped
    sigma = sigma @ tf.transpose(_u)
    # normalize it
    W_bar = W_reshaped / sigma
    return W_bar, _u, sigma
