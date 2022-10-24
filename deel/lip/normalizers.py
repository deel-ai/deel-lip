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

DEFAULT_MAX_NITER_BJORCK = 15
DEFAULT_BETA_BJORCK = 0.5
DEFAULT_EPS_SPECTRAL = 1e-3
DEFAULT_EPS_BJORCK = 1e-3


def reshaped_kernel_orthogonalization(
    kernel,
    u,
    adjustment_coef,
    eps_spectral=DEFAULT_EPS_SPECTRAL,
    eps_bjorck=DEFAULT_EPS_BJORCK,
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
        eps_spectral: stopping criterion in spectral algorithm
        eps_bjorck: stopping criterion in bjorck algorithm
        beta: the beta used in the bjorck algorithm

    Returns: the orthogonalized kernel, the new u, and sigma which is the largest
        singular value

    """
    W_bar, u, sigma = spectral_normalization(kernel, u, eps=eps_spectral)
    if (eps_bjorck is not None) and (beta is not None):
        W_bar = bjorck_normalization(W_bar, eps=eps_bjorck, beta=beta)
    W_bar = W_bar * adjustment_coef
    W_bar = K.reshape(W_bar, kernel.shape)
    return W_bar, u, sigma


def bjorck_normalization(w, eps=DEFAULT_EPS_BJORCK, beta=DEFAULT_BETA_BJORCK):
    """
    apply Bjorck normalization on w.

    Args:
        w: weight to normalize, in order to work properly, we must have
            max_eigenval(w) ~= 1
        eps: epsilon stopping criterion: norm(wt - wt-1) must be less than eps
        beta: beta used in each iteration, must be in the interval ]0, 0.5]

    Returns:
        the orthonormal weights

    """
    # create a fake old_w that does'nt pass the loop condition
    # it won't affect computation as the first action done in the loop overwrite it.
    old_w = 10 * w
    # define the loop condition

    def cond(w, old_w):
        return tf.linalg.norm(w - old_w) >= eps

    # define the loop body
    def body(w, old_w):
        old_w = w
        w = (1 + beta) * w - beta * w @ tf.transpose(w) @ w
        return w, old_w

    # apply the loop
    w, old_w = tf.while_loop(
        cond, body, (w, old_w), parallel_iterations=1, maximum_iterations=DEFAULT_MAX_NITER_BJORCK
    )
    return w


def _power_iteration(w, u, eps=DEFAULT_EPS_SPECTRAL):
    """
    Internal function that performs the power iteration algorithm.

    Args:
        w: weights matrix that we want to find eigen vector
        u: initialization of the eigen vector
        eps: epsilon stopping criterion: norm(ut - ut-1) must be less than eps

    Returns:
         u and v corresponding to the maximum eigenvalue

    """
    # build _u and _v (_v is size of _u@tf.transpose(w), will be set on the first body
    # iteration)
    _u = u
    _v = tf.zeros(u.shape[:-1] + (w.shape[0],), dtype=w.dtype)

    # create a fake old_w that doesn't pass the loop condition
    # it won't affect computation as the first action done in the loop overwrite it.
    _old_u = 10 * _u

    # define the loop condition
    def cond(_u, _v, old_u):
        return tf.linalg.norm(_u - old_u) >= eps

    # define the loop body
    def body(_u, _v, _old_u):
        _old_u = _u
        _v = tf.math.l2_normalize(_u @ tf.transpose(w))
        _u = tf.math.l2_normalize(_v @ w)
        return _u, _v, _old_u

    # apply the loop
    _u, _v, _old_u = tf.while_loop(
        cond, body, (_u, _v, _old_u), parallel_iterations=1, maximum_iterations=DEFAULT_MAX_NITER_BJORCK
    )
    return _u, _v


def spectral_normalization(kernel, u, eps=DEFAULT_EPS_SPECTRAL):
    """
    Normalize the kernel to have it's max eigenvalue == 1.

    Args:
        kernel: the kernel to normalize
        u: initialization for the max eigen vector
        eps: epsilon stopping criterion: norm(ut - ut-1) must be less than eps

    Returns:
        the normalized kernel w_bar, it's shape, the maximum eigen vector, and the
        maximum singular value

    """
    W_shape = kernel.shape
    if u is None:
        u = tf.linalg.l2_normalize(tf.ones(shape=tuple([1, W_shape[-1]])))
    # Flatten the Tensor
    W_reshaped = tf.reshape(kernel, [-1, W_shape[-1]])
    _u, _v = _power_iteration(W_reshaped, u, eps)
    # compute Sigma
    sigma = _v @ W_reshaped
    sigma = sigma @ tf.transpose(_u)
    # normalize it
    # we assume that in the worst case we converged to sigma + eps (as u and v are
    # normalized after each iteration)
    # in order to be sure that operator norm of W_bar is strictly less than one we
    # use sigma + eps, which ensure stability of the bjorck even when beta=0.5
    W_bar = W_reshaped / (sigma + eps)
    return W_bar, _u, sigma
