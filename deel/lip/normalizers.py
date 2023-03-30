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

from .utils import _maybe_transpose_kernel, _zero_upscale2D

DEFAULT_BETA_BJORCK = 0.5
DEFAULT_EPS_SPECTRAL = 1e-3
DEFAULT_EPS_BJORCK = 1e-3
DEFAULT_MAXITER_BJORCK = 15
DEFAULT_MAXITER_SPECTRAL = 10
SWAP_MEMORY = True
STOP_GRAD_SPECTRAL = True
GRAD_PASSTHROUGH_BJORCK = False


def set_swap_memory(value: bool):
    """
    Set the global SWAP_MEMORY to values. This function must be called before
    constructing the model (first call of `reshaped_kernel_orthogonalization`) in
    order to be accounted.

    Args:
        value: boolean that will be used as the swap_memory parameter in while loops
            in spectral and bjorck algorithms.

    """
    global SWAP_MEMORY
    SWAP_MEMORY = value


def set_stop_grad_spectral(value: bool):
    """
    Set the global STOP_GRAD_SPECTRAL to values. This function must be called before
    constructing the model (first call of `reshaped_kernel_orthogonalization`) in
    order to be accounted.

    Args:
        value: boolean, when set to True, disable back-propagation through the power
            iteration algorithm. The back-propagation will account how updates affects
            the maximum singular value but not how it affects the largest singular
            vector. When set to False, back-propagate through the while loop.

    """
    global STOP_GRAD_SPECTRAL
    STOP_GRAD_SPECTRAL = value


def set_grad_passthrough_bjorck(value: bool):
    """
    Set the global GRAD_PASSTHROUGH_BJORCK to value. This function must be called before
    constructing the model (first call of `reshaped_kernel_orthogonalization`) in
    order to be accounted.

    Args:
        value: boolean, when set to True, only back-propagate through a single bjorck
            iteration. In other words it will act with the usual variable number of
            iterations during forward but will have bjorck_niter=1 during backward.
            This allows to save time and memory. When set to False, back-propagate
            through the bjorck loop.

    """
    global GRAD_PASSTHROUGH_BJORCK
    GRAD_PASSTHROUGH_BJORCK = value


def _check_RKO_params(eps_spectral, eps_bjorck, beta_bjorck):
    """Assert that the RKO hyper-parameters are supported values."""
    if eps_spectral <= 0:
        raise ValueError("eps_spectral has to be > 0")
    if (eps_bjorck is not None) and (eps_bjorck <= 0.0):
        raise ValueError("eps_bjorck must be > 0")
    if (beta_bjorck is not None) and not (0.0 < beta_bjorck <= 0.5):
        raise ValueError("beta_bjorck must be in ]0, 0.5]")


def reshaped_kernel_orthogonalization(
    kernel,
    u,
    adjustment_coef,
    eps_spectral=DEFAULT_EPS_SPECTRAL,
    eps_bjorck=DEFAULT_EPS_BJORCK,
    beta=DEFAULT_BETA_BJORCK,
    maxiter_spectral=DEFAULT_MAXITER_SPECTRAL,
    maxiter_bjorck=DEFAULT_MAXITER_BJORCK,
):
    """
    Perform reshaped kernel orthogonalization (RKO) to the kernel given as input. It
    apply the power method to find the largest singular value and apply the Bjorck
    algorithm to the rescaled kernel. This greatly improve the stability and and
    speed convergence of the bjorck algorithm.

    Args:
        kernel (tf.Tensor): the kernel to orthogonalize
        u (tf.Tensor): the vector used to do the power iteration method
        adjustment_coef (float): the adjustment coefficient as used in convolution
        eps_spectral (float): stopping criterion in spectral algorithm
        eps_bjorck (float): stopping criterion in bjorck algorithm
        beta (float): the beta used in the bjorck algorithm
        maxiter_spectral (int): maximum number of iterations for the power iteration
        maxiter_bjorck (int): maximum number of iterations for bjorck algorithm

    Returns:
        tf.Tensor: the orthogonalized kernel, the new u, and sigma which is the largest
            singular value

    """
    W_shape = kernel.shape
    # Flatten the Tensor
    W_reshaped = tf.reshape(kernel, [-1, W_shape[-1]])
    W_bar, u, sigma = spectral_normalization(
        W_reshaped, u, eps=eps_spectral, maxiter=maxiter_spectral
    )
    if (eps_bjorck is not None) and (beta is not None):
        W_bar = bjorck_normalization(
            W_bar, eps=eps_bjorck, beta=beta, maxiter=maxiter_bjorck
        )
    W_bar = W_bar * adjustment_coef
    W_bar = K.reshape(W_bar, kernel.shape)
    return W_bar, u, sigma


def _wwtw(w):
    if w.shape[0] > w.shape[1]:
        return w @ (tf.transpose(w) @ w)
    else:
        return (w @ tf.transpose(w)) @ w


def bjorck_normalization(
    w, eps=DEFAULT_EPS_BJORCK, beta=DEFAULT_BETA_BJORCK, maxiter=DEFAULT_MAXITER_BJORCK
):
    """
    apply Bjorck normalization on w.

    Args:
        w (tf.Tensor): weight to normalize, in order to work properly, we must have
            max_eigenval(w) ~= 1
        eps (float): epsilon stopping criterion: norm(wt - wt-1) must be less than eps
        beta (float): beta used in each iteration, must be in the interval ]0, 0.5]
        maxiter (int): maximum number of iterations for the algorithm

    Returns:
        tf.Tensor: the orthonormal weights

    """
    # create a fake old_w that does'nt pass the loop condition
    # it won't affect computation as the first action done in the loop overwrite it.
    old_w = w
    w = (1 + beta) * w - beta * _wwtw(w)
    # define the loop condition

    def cond(w, old_w):
        return tf.linalg.norm(w - old_w) >= eps

    # define the loop body
    def body(w, old_w):
        old_w = w
        w = (1 + beta) * w - beta * _wwtw(w)
        return w, old_w

    # apply the loop
    def while_func(w, old_w):
        return tf.while_loop(
        cond,
        body,
        (w, old_w),
        parallel_iterations=30,
        maximum_iterations=maxiter,
        swap_memory=SWAP_MEMORY,
        )
    if GRAD_PASSTHROUGH_BJORCK:
        w, old_w = tf.grad_pass_through(while_func)(w, old_w)
    else:
        w, old_w = while_func(w, old_w)
    return w


def _power_iteration(w, u, eps=DEFAULT_EPS_SPECTRAL, maxiter=DEFAULT_MAXITER_SPECTRAL):
    """
    Internal function that performs the power iteration algorithm.

    Args:
        w: weights matrix that we want to find eigen vector
        u: initialization of the eigen vector
        eps: epsilon stopping criterion: norm(ut - ut-1) must be less than eps
        maxiter: maximum number of iterations for the algorithm

    Returns:
         u and v corresponding to the maximum eigenvalue

    """
    # build _u and _v (_v is size of _u@tf.transpose(w), will be set on the first body
    # iteration)
    if u is None:
        u = tf.linalg.l2_normalize(
            tf.random.uniform(
                shape=(1, w.shape[-1]), minval=0.0, maxval=1.0, dtype=w.dtype
            )
        )
    _u = u
    _v = tf.zeros((1,) + (w.shape[0],), dtype=w.dtype)

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
        cond,
        body,
        (_u, _v, _old_u),
        parallel_iterations=30,
        maximum_iterations=maxiter,
        swap_memory=SWAP_MEMORY,
    )
    if STOP_GRAD_SPECTRAL:
        _u = tf.stop_gradient(_u)
        _v = tf.stop_gradient(_v)
    return _u, _v


def spectral_normalization(
    kernel, u, eps=DEFAULT_EPS_SPECTRAL, maxiter=DEFAULT_MAXITER_SPECTRAL
):
    """
    Normalize the kernel to have it's max eigenvalue == 1.

    Args:
        kernel (tf.Tensor): the kernel to normalize, assuming a 2D kernel
        u (tf.Tensor): initialization for the max eigen vector
        eps (float): epsilon stopping criterion: norm(ut - ut-1) must be less than eps
        maxiter (int): maximum number of iterations for the algorithm

    Returns:
        the normalized kernel w_bar, the maximum eigen vector, and the maximum singular
            value.

    """
    _u, _v = _power_iteration(kernel, u, eps, maxiter)
    # compute Sigma
    sigma = _v @ kernel
    sigma = sigma @ tf.transpose(_u)
    # normalize it
    # we assume that in the worst case we converged to sigma + eps (as u and v are
    # normalized after each iteration)
    # in order to be sure that operator norm of W_bar is strictly less than one we
    # use sigma + eps, which ensure stability of the bjorck even when beta=0.5
    W_bar = kernel / (sigma + eps)
    return W_bar, _u, sigma


def _power_iteration_conv(
    w,
    u,
    stride=1.0,
    conv_first=True,
    pad_func=None,
    eps=DEFAULT_EPS_SPECTRAL,
    maxiter=DEFAULT_MAXITER_SPECTRAL,
    big_constant=-1,
):
    """
    Internal function that performs the power iteration algorithm for convolution.

    Args:
        w: weights matrix that we want to find eigen vector
        u: initialization of the eigen matrix should be ||u||=1 for L2_norm
        stride: stride parameter of the convolution
        conv_first: RO or CO case , should be True in CO case (stride^2*C<M)
        pad_func: function for applying padding (None is padding same)
        eps: epsilon stopping criterion: norm(ut - ut-1) must be less than eps
        maxiter: maximum number of iterations for the algorithm
        big_constant: only for computing the minimum singular value (otherwise -1)
    Returns:
         u and v corresponding to the maximum eigenvalue

    """

    def identity(x):
        return x

    # If pad_func is None, standard convolution with SAME padding
    # Else, pad_func padding function (externally defined)
    #       + standard convolution with VALID padding.
    if pad_func is None:
        pad_type = "SAME"
        _pad_func = identity
    else:
        pad_type = "VALID"
        _pad_func = pad_func

    def _conv(u, w, stride):
        u_pad = _pad_func(u)
        return tf.nn.conv2d(u_pad, w, stride, pad_type)

    def _conv_transpose(u, w, output_shape, stride):
        if pad_func is None:
            return tf.nn.conv2d_transpose(u, w, output_shape, stride, pad_type)
        else:
            u_upscale = _zero_upscale2D(u, (stride, stride))
            w_adj = _maybe_transpose_kernel(w, True)
            return _conv(u_upscale, w_adj, stride=1)

    def body(_u, _v, _old_u, _norm_u):
        # _u is supposed to be normalized when entering in the body function
        _old_u = _u
        u = _u

        if conv_first:  # Conv, then transposed conv
            v = _conv(u, w, stride)
            unew = _conv_transpose(v, w, u.shape, stride)
        else:  # Transposed conv, then conv
            v = _conv_transpose(u, w, _v.shape, stride)
            unew = _conv(v, w, stride)

        if big_constant > 0:
            unew = big_constant * u - unew

        _norm_unew = tf.norm(unew)
        unew = tf.math.l2_normalize(unew)
        return unew, v, _old_u, _norm_unew

    # define the loop condition

    def cond(_u, _v, old_u, _norm_u):
        return tf.linalg.norm(_u - old_u) >= eps

    # v shape
    if conv_first:
        v_shape = (
            (u.shape[0],)
            + (u.shape[1] // stride, u.shape[2] // stride)
            + (w.shape[-1],)
        )
    else:
        v_shape = (
            (u.shape[0],) + (u.shape[1] * stride, u.shape[2] * stride) + (w.shape[-2],)
        )

    # build _u and _v
    _norm_u = tf.norm(u)
    _u = tf.math.l2_normalize(u)
    _u += tf.random.uniform(_u.shape, minval=-eps, maxval=eps)
    _v = tf.zeros(v_shape)  # _v will be set on the first body iteration

    # create a fake old_w that doesn't pass the loop condition
    # it won't affect computation as the first action done in the loop overwrites it.
    _old_u = 10 * _u

    # apply the loop
    _u, _v, _old_u, _norm_u = tf.while_loop(
        cond,
        body,
        (_u, _v, _old_u, _norm_u),
        parallel_iterations=1,
        maximum_iterations=maxiter,
        swap_memory=SWAP_MEMORY,
    )
    if STOP_GRAD_SPECTRAL:
        _u = tf.stop_gradient(_u)
        _v = tf.stop_gradient(_v)

    return _u, _v, _norm_u


@tf.function
def spectral_normalization_conv(
    kernel,
    u,
    stride=1.0,
    conv_first=True,
    pad_func=None,
    eps=DEFAULT_EPS_SPECTRAL,
    maxiter=DEFAULT_MAXITER_SPECTRAL,
):
    """
    Normalize the convolution kernel to have its max eigenvalue == 1.

    Args:
        kernel (tf.Tensor): the convolution kernel to normalize
        u (tf.Tensor): initialization for the max eigen vector (as a 4d tensor)
        stride (int): stride parameter of convolutions
        conv_first (bool): RO or CO case , should be True in CO case (stride^2*C<M)
        pad_func (Callable): function for applying padding (None is padding same)
        eps (float): epsilon stopping criterion: norm(ut - ut-1) must be less than eps
        maxiter (int): maximum number of iterations for the power iteration algorithm.

    Returns:
        the normalized kernel w_bar, the maximum eigen vector, and the maximum eigen
            value
    """

    if eps < 0:
        return kernel, u, 1.0

    _u, _v, _ = _power_iteration_conv(
        kernel, u, stride, conv_first, pad_func, eps, maxiter
    )

    # Calculate Sigma
    sigma = tf.norm(_v)
    W_bar = kernel / (sigma + eps)
    return W_bar, _u, sigma


def _power_iteration_dw_conv(
    w,
    u,
    strides=(1, 1),
    pad_func=None,
    filterwise=False,
    eps=DEFAULT_EPS_SPECTRAL,
    maxiter=DEFAULT_MAXITER_SPECTRAL,
    big_constant=-1,
):
    """
    Internal function that performs the power iteration algorithm for convolution.

    Args:
        w: weights matrix that we want to find eigen vector
        u: initialization of the eigen matrix should be ||u||=1 for L2_norm
        stride: strides parameter of the convolution
        conv_first: RO or CO case , should be True in CO case (strides^2*C<M)
        pad_func: function for applying padding (None is padding same)
        eps: epsilon stopping criterion: norm(ut - ut-1) must be less than eps
        maxiter: maximum number of iterations for the algorithm
        big_constant: only for computing the minimum singular value (otherwise -1)
    Returns:
         u and v corresponding to the maximum eigenvalue

    """

    def identity(x):
        return x

    norm_axes = (0, 1) if filterwise else None

    # If pad_func is None, standard convolution with SAME padding
    # Else, pad_func padding function (externally defined)
    #       + standard convolution with VALID padding.
    if pad_func is None:
        pad_type = "SAME"
        _pad_func = identity
    else:
        pad_type = "VALID"
        _pad_func = pad_func

    def _dw_conv(u, w, strides):
        u_pad = _pad_func(u)
        # non working with channels_first format when stride != 1
        return tf.nn.depthwise_conv2d(
            u_pad, w, (1, strides[0], strides[1], 1), pad_type
        )

    def _dw_conv_transpose(u, w, strides):
        u_upscale = _zero_upscale2D(u, strides)
        w_adj = w[::-1, ::-1, :]
        return _dw_conv(u_upscale, w_adj, strides=(1, 1))

    def body(_u, _v, _old_u, _norm_u):
        # _u is supposed to be normalized when entering in the body function
        _old_u = _u
        u = _u

        v = _dw_conv(u, w, strides)
        unew = _dw_conv_transpose(v, w, strides)

        if big_constant > 0:
            unew = big_constant * u - unew

        _norm_unew = tf.norm(unew)
        unew = tf.math.l2_normalize(unew, axis=norm_axes)
        return unew, v, _old_u, _norm_unew

    # define the loop condition

    def cond(_u, _v, old_u, _norm_u):
        return tf.linalg.norm(_u - old_u) >= eps

    # v shape
    v_shape = (
        (u.shape[0],)
        + (u.shape[1] // strides[0], u.shape[2] // strides[1])
        + (u.shape[-1],)
    )

    # build _u and _v
    _norm_u = tf.norm(u)
    _u = tf.math.l2_normalize(u, axis=norm_axes)
    _u += tf.random.uniform(_u.shape, minval=-eps, maxval=eps)
    _v = tf.zeros(v_shape)  # _v will be set on the first body iteration

    # create a fake old_w that doesn't pass the loop condition
    # it won't affect computation as the first action done in the loop overwrites it.
    _old_u = 10 * _u

    # apply the loop
    _u, _v, _old_u, _norm_u = tf.while_loop(
        cond,
        body,
        (_u, _v, _old_u, _norm_u),
        parallel_iterations=1,
        maximum_iterations=maxiter,
        swap_memory=SWAP_MEMORY,
    )
    if STOP_GRAD_SPECTRAL:
        _u = tf.stop_gradient(_u)
        _v = tf.stop_gradient(_v)

    return _u, _v, _norm_u


@tf.function
def spectral_normalization_dw_conv(
    kernel,
    u,
    stride=1.0,
    pad_func=None,
    filterwise=False,
    eps=DEFAULT_EPS_SPECTRAL,
    maxiter=DEFAULT_MAXITER_SPECTRAL,
):
    """
    Normalize the convolution kernel to have its max eigenvalue == 1.

    Args:
        kernel (tf.Tensor): the convolution kernel to normalize
        u (tf.Tensor): initialization for the max eigen vector (as a 4d tensor)
        stride (int): stride parameter of convolutions
        pad_func (Callable): function for applying padding (None is padding same)
        eps (float): epsilon stopping criterion: norm(ut - ut-1) must be less than eps
        maxiter (int): maximum number of iterations for the power iteration algorithm.

    Returns:
        the normalized kernel w_bar, the maximum eigen vector, and the maximum eigen
            value
    """

    if eps < 0:
        return kernel, u, 1.0

    _u, _v, _ = _power_iteration_dw_conv(
        kernel, u, stride, pad_func, filterwise, eps, maxiter
    )

    # Calculate Sigma
    axes = (0, 1, 2) if filterwise else None
    sigma = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(u), axis=axes)), (1, 1, -1, 1))
    W_bar = kernel / (sigma + eps)
    return W_bar, _u, sigma
