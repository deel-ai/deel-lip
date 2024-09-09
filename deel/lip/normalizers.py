# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains computation function, for Bjorck and spectral
normalization. This is done for internal use only.
"""
import keras
import keras.ops as K

from .utils import _maybe_transpose_kernel, _zero_upscale2D, l2_normalize

DEFAULT_BETA_BJORCK = 0.5
DEFAULT_EPS_SPECTRAL = 1e-3
DEFAULT_EPS_BJORCK = 1e-3
DEFAULT_MAXITER_BJORCK = 15
DEFAULT_MAXITER_SPECTRAL = 10
STOP_GRAD_SPECTRAL = True


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
        kernel (Tensor): the kernel to orthogonalize
        u (Tensor): the vector used to do the power iteration method
        adjustment_coef (float): the adjustment coefficient as used in convolution
        eps_spectral (float): stopping criterion in spectral algorithm
        eps_bjorck (float): stopping criterion in bjorck algorithm
        beta (float): the beta used in the bjorck algorithm
        maxiter_spectral (int): maximum number of iterations for the power iteration
        maxiter_bjorck (int): maximum number of iterations for bjorck algorithm

    Returns:
        Tensor: the orthogonalized kernel, the new u, and sigma which is the largest
            singular value

    """
    W_shape = kernel.shape
    # Flatten the Tensor
    W_reshaped = K.reshape(kernel, [-1, W_shape[-1]])
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
        return w @ (K.transpose(w) @ w)
    else:
        return (w @ K.transpose(w)) @ w


def bjorck_normalization(
    w, eps=DEFAULT_EPS_BJORCK, beta=DEFAULT_BETA_BJORCK, maxiter=DEFAULT_MAXITER_BJORCK
):
    """
    apply Bjorck normalization on w.

    Args:
        w (Tensor): weight to normalize, in order to work properly, we must have
            max_eigenval(w) ~= 1
        eps (float): epsilon stopping criterion: norm(wt - wt-1) must be less than eps
        beta (float): beta used in each iteration, must be in the interval ]0, 0.5]
        maxiter (int): maximum number of iterations for the algorithm

    Returns:
        Tensor: the orthonormal weights

    """
    # create a fake old_w that does'nt pass the loop condition
    # it won't affect computation as the first action done in the loop overwrite it.
    old_w = 10 * w
    # define the loop condition

    def cond(w, old_w):
        return K.norm(w - old_w) >= eps

    # define the loop body
    def body(w, old_w):
        old_w = w
        w = (1 + beta) * w - beta * _wwtw(w)
        return w, old_w

    # apply the loop
    w, old_w = K.while_loop(
        cond,
        body,
        (w, old_w),
        maximum_iterations=maxiter,
    )
    return w


def _power_iteration(
    linear_operator,
    adjoint_operator,
    u,
    eps=DEFAULT_EPS_SPECTRAL,
    maxiter=DEFAULT_MAXITER_SPECTRAL,
    axis=None,
):
    """Internal function that performs the power iteration algorithm to estimate the
    largest singular vector of a linear operator.

    Args:
        linear_operator (Callable): a callable object that maps a linear operation.
        adjoint_operator (Callable): a callable object that maps the adjoint of the
            linear operator.
        u (Tensor): initialization of the singular vector.
        eps (float, optional): stopping criterion of the algorithm, when
            norm(u[t] - u[t-1]) is less than eps. Defaults to DEFAULT_EPS_SPECTRAL.
        maxiter (int, optional): maximum number of iterations for the algorithm.
            Defaults to DEFAULT_MAXITER_SPECTRAL.
        axis (int/list, optional): dimension along which to normalize. Can be set for
            depthwise convolution for example. Defaults to None.

    Returns:
        Tensor: the maximum singular vector.
    """

    # Prepare while loop variables
    u = l2_normalize(u, axis=axis)
    # create a fake old_w that doesn't pass the loop condition, it will be overwritten
    old_u = u + 2 * eps

    # Loop body
    def body(u, old_u):
        old_u = u
        v = linear_operator(u)
        u = adjoint_operator(v)

        u = l2_normalize(u, axis=axis)

        return u, old_u

    # Loop stopping condition
    def cond(u, old_u):
        return K.norm(K.reshape(u, [-1]) - K.reshape(old_u, [-1])) >= eps

    # Run the while loop
    u, _ = K.while_loop(
        cond,
        body,
        (u, old_u),
        maximum_iterations=maxiter,
    )

    # Prevent gradient to back-propagate into the while loop
    if STOP_GRAD_SPECTRAL:
        u = K.stop_gradient(u)

    return u


def spectral_normalization(
    kernel, u, eps=DEFAULT_EPS_SPECTRAL, maxiter=DEFAULT_MAXITER_SPECTRAL
):
    """
    Normalize the kernel to have its maximum singular value equal to 1.

    Args:
        kernel (Tensor): the kernel to normalize, assuming a 2D kernel.
        u (Tensor): initialization of the maximum singular vector.
        eps (float, optional): stopping criterion of the algorithm, when
            norm(u[t] - u[t-1]) is less than eps. Defaults to DEFAULT_EPS_SPECTRAL.
        maxiter (int, optional): maximum number of iterations for the algorithm.
            Defaults to DEFAULT_MAXITER_SPECTRAL.

    Returns:
        the normalized kernel, the maximum singular vector, and the maximum singular
            value.
    """

    if u is None:
        u = keras.random.uniform(
            shape=(1, kernel.shape[-1]), minval=0.0, maxval=1.0, dtype=kernel.dtype
        )

    def linear_op(u):
        return u @ K.transpose(kernel)

    def adjoint_op(v):
        return v @ kernel

    u = _power_iteration(linear_op, adjoint_op, u, eps, maxiter)

    # Compute the largest singular value and the normalized kernel.
    # We assume that in the worst case we converged to sigma + eps (as u and v are
    # normalized after each iteration)
    # In order to be sure that operator norm of normalized kernel is strictly less than
    # one we use sigma + eps, which ensures stability of Björck algorithm even when
    # beta=0.5
    sigma = K.reshape(K.norm(linear_op(u)), (1, 1))
    normalized_kernel = kernel / (sigma + eps)
    return normalized_kernel, u, sigma


def get_conv_operators(kernel, u_shape, stride=1.0, conv_first=True, pad_func=None):
    """
    Return two functions corresponding to the linear convolution operator and its
    adjoint.

    Args:
        kernel (Tensor): the convolution kernel to normalize
        u_shape (tuple): shape of a singular vector (as a 4D tensor).
        stride (int, optional): stride parameter of convolutions. Defaults to 1.
        conv_first (bool, optional): RO or CO case , should be True in CO case
            (stride^2*C<M). Defaults to True.
        pad_func (Callable, optional): function for applying padding (None is padding
            same). Defaults to None.

    Returns:
        tuple: two functions for the linear convolution operator and its adjoint
            operator.
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
        return K.conv(u_pad, w, stride, pad_type)

    def _conv_transpose(u, w, output_shape, stride):
        if pad_func is None:
            return K.conv_transpose(u, w, output_shape, stride, pad_type)
        else:
            u_upscale = _zero_upscale2D(u, (stride, stride))
            w_adj = _maybe_transpose_kernel(w, True)
            return _conv(u_upscale, w_adj, stride=1)

    if conv_first:

        def linear_op(u):
            return _conv(u, kernel, stride)

        def adjoint_op(v):
            return _conv_transpose(v, kernel, u_shape, stride)

    else:
        v_shape = (
            (u_shape[0],)
            + (u_shape[1] * stride, u_shape[2] * stride)
            + (kernel.shape[-2],)
        )

        def linear_op(u):
            return _conv_transpose(u, kernel, v_shape, stride)

        def adjoint_op(v):
            return _conv(v, kernel, stride)

    return linear_op, adjoint_op


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
        kernel (Tensor): the convolution kernel to normalize
        u (Tensor): initialization for the max eigen vector (as a 4d tensor)
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

    linear_op, adjoint_op = get_conv_operators(
        kernel, u.shape, stride, conv_first, pad_func
    )

    u = l2_normalize(u) + keras.random.uniform(u.shape, minval=-eps, maxval=eps)
    u = _power_iteration(linear_op, adjoint_op, u, eps, maxiter)

    # Compute the largest singular value and the normalized kernel
    sigma = K.norm(K.reshape(linear_op(u), [-1]))
    normalized_kernel = kernel / (sigma + eps)
    return normalized_kernel, u, sigma
