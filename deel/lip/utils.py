# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Contains utility functions.
"""
from typing import Generator, Tuple, Any
import numpy as np
import keras
import keras.ops as K


def l2_normalize(x, axis=None, epsilon=1e-12):
    """
    Normalizes a tensor wrt the L2 norm alongside the specified axis.

    Inspired by `tf.math.l2_normalize()` function.

    Args:
        x: Tensor.
        axis (optional): Dimension along which to normalize. A scalar or a vector of
            integers. Defaults to None (all axes considered).
        epsilon (optional): small value to avoid division by zero. Defaults to 1e-12.

    Returns:
        Tensor: normalized tensor
    """
    square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
    x_inv_norm = K.rsqrt(K.maximum(square_sum, epsilon))
    return K.multiply(x, x_inv_norm)


def evaluate_lip_const_gen(
    model: keras.Model,
    generator: Generator[Tuple[np.ndarray, np.ndarray], Any, None],
    eps=1e-4,
    seed=None,
):
    """
    Evaluate the Lipschitz constant of a model, with the naive method.
    Please note that the estimation of the lipschitz constant is done locally around
    input sample. This may not correctly estimate the behaviour in the whole domain.
    The computation might also be inaccurate in high dimensional space.

    This is the generator version of evaluate_lip_const.

    Args:
        model: built keras model used to make predictions
        generator: used to select datapoints where to compute the lipschitz constant
        eps (float): magnitude of noise to add to input in order to compute the constant
        seed (int): seed used when generating the noise ( can be set to None )

    Returns:
        float: the empirically evaluated lipschitz constant.

    """
    x, _ = generator.send(None)
    return evaluate_lip_const(model, x, eps, seed=seed)


def evaluate_lip_const(model: keras.Model, x, eps=1e-4, seed=None):
    """
    Evaluate the Lipschitz constant of a model, with the naive method.
    Please note that the estimation of the lipschitz constant is done locally around
    input sample. This may not correctly estimate the behaviour in the whole domain.

    Args:
        model: built keras model used to make predictions
        x: inputs used to compute the lipschitz constant
        eps (float): magnitude of noise to add to input in order to compute the constant
        seed (int): seed used when generating the noise ( can be set to None )

    Returns:
        float: the empirically evaluated lipschitz constant. The computation might also
            be inaccurate in high dimensional space.

    """
    y_pred = model.predict(x)
    # x = np.repeat(x, 100, 0)
    # y_pred = np.repeat(y_pred, 100, 0)
    x_var = x + keras.random.uniform(
        shape=x.shape, minval=eps * 0.25, maxval=eps, seed=seed
    )
    y_pred_var = model.predict(x_var)
    dx = x - x_var
    dfx = y_pred - y_pred_var
    ndx = K.sqrt(K.sum(K.square(dx), axis=range(1, len(x.shape))))
    ndfx = K.sqrt(K.sum(K.square(dfx), axis=range(1, len(y_pred.shape))))
    lip_cst = K.max(ndfx / ndx)
    print(f"lip cst: {lip_cst:.3f}")
    return lip_cst


def _padding_circular(x, circular_paddings):
    """Add circular padding to a 4-D tensor. Only channels_last is supported."""
    if circular_paddings is None:
        return x
    w_pad, h_pad = circular_paddings
    if w_pad > 0:
        x = K.concatenate((x[:, -w_pad:, :, :], x, x[:, :w_pad, :, :]), axis=1)
    if h_pad > 0:
        x = K.concatenate((x[:, :, -h_pad:, :], x, x[:, :, :h_pad, :]), axis=2)
    return x


def _zero_upscale2D(x, strides):
    stride_v = strides[0] * strides[1]
    if stride_v == 1:
        return x
    output_shape = x.get_shape().as_list()[1:]
    if strides[1] > 1:
        output_shape[1] *= strides[1]
        x = K.expand_dims(x, 3)
        fillz = K.zeros_like(x)
        fillz = K.tile(fillz, [1, 1, 1, strides[1] - 1, 1])
        x = K.concatenate((x, fillz), axis=3)
        x = K.reshape(x, (-1,) + tuple(output_shape))
    if strides[0] > 1:
        output_shape[0] *= strides[0]
        x = K.expand_dims(x, 2)
        fillz = K.zeros_like(x)
        fillz = K.tile(fillz, [1, 1, strides[0] - 1, 1, 1])
        x = K.concatenate((x, fillz), axis=2)
        x = K.reshape(x, (-1,) + tuple(output_shape))
    return x


def _maybe_transpose_kernel(w, transpose=False):
    """Transpose 4-D kernel: permutation of axes 2 and 3 + reverse axes 0 and 1."""
    if not transpose:
        return w
    w_adj = K.transpose(w, axes=[0, 1, 3, 2])
    w_adj = w_adj[::-1, ::-1, :]
    return w_adj


def process_labels_for_multi_gpu(labels):
    """Process labels to be fed to any loss based on KR estimation with a multi-GPU/TPU
    strategy.

    When using a multi-GPU/TPU strategy, the flag `multi_gpu` in KR-based losses must be
    set to True and the labels have to be pre-processed with this function.

    For binary classification, the labels should be of shape [batch_size, 1].
    For multiclass problems, the labels must be one-hot encoded (1 or 0) with shape
    [batch_size, number of classes].

    Args:
        labels (Tensor): tensor containing the labels

    Returns:
        Tensor: labels processed for KR-based losses with multi-GPU/TPU strategy.
    """
    eps = 1e-7
    labels = K.cast(K.where(labels > 0, 1, 0), labels.dtype)
    batch_size = K.cast(K.shape(labels)[0], labels.dtype)
    counts = K.sum(labels, axis=0)

    pos = labels / (counts + eps)
    neg = (1 - labels) / (batch_size - counts + eps)
    # Since element-wise KR terms are averaged by loss reduction later on, it is needed
    # to multiply by batch_size here.
    return batch_size * (pos - neg)
