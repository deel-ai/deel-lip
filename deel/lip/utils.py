# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Contains utility functions.
"""
from typing import Generator, Tuple, Any
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model


def evaluate_lip_const_gen(
    model: Model, generator: Generator[Tuple[np.ndarray, np.ndarray], Any, None]
):
    """
    Evaluate the Lipschitz constant of a model, using the Jacobian of the model.
    Please note that the estimation of the Lipschitz constant is done locally around
    input samples. This may not correctly estimate the behaviour in the whole domain.
    The computation might also be inaccurate in high dimensional space.

    This is the generator version of evaluate_lip_const.

    Args:
        model: built keras model used to make predictions
        generator: used to select datapoints where to compute the lipschitz constant

    Returns:
        float: the empirically evaluated lipschitz constant.

    """
    x, _ = generator.send(None)
    return evaluate_lip_const(model, x)


def evaluate_lip_const(model: Model, x):
    """
    Evaluate the Lipschitz constant of a model, using the Jacobian of the model.
    Please note that the estimation of the lipschitz constant is done locally around
    input samples. This may not correctly estimate the behaviour in the whole domain.

    Args:
        model: built keras model used to make predictions
        x: inputs used to compute the lipschitz constant

    Returns:
        float: the empirically evaluated Lipschitz constant. The computation might also
            be inaccurate in high dimensional space.

    """
    batch_size = x.shape[0]
    x = tf.constant(x, dtype=model.input.dtype)

    # Get the jacobians of the model w.r.t. the inputs
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_pred = model(x, training=False)
    batch_jacobian = tape.batch_jacobian(y_pred, x)

    # Reshape the jacobians (in case of multi-dimensional input/output like in conv)
    xdim = tf.reduce_prod(x.shape[1:])
    ydim = tf.reduce_prod(y_pred.shape[1:])
    batch_jacobian = tf.reshape(batch_jacobian, (batch_size, ydim, xdim))

    # Compute the spectral norm of the jacobians and return the maximum
    b = tf.norm(batch_jacobian, ord=2, axis=[-2, -1]).numpy()
    return tf.reduce_max(b)


def _padding_circular(x, circular_paddings):
    """Add circular padding to a 4-D tensor. Only channels_last is supported."""
    if circular_paddings is None:
        return x
    w_pad, h_pad = circular_paddings
    if w_pad > 0:
        x = tf.concat((x[:, -w_pad:, :, :], x, x[:, :w_pad, :, :]), axis=1)
    if h_pad > 0:
        x = tf.concat((x[:, :, -h_pad:, :], x, x[:, :, :h_pad, :]), axis=2)
    return x


def _zero_upscale2D(x, strides):
    stride_v = strides[0] * strides[1]
    if stride_v == 1:
        return x
    output_shape = x.get_shape().as_list()[1:]
    if strides[1] > 1:
        output_shape[1] *= strides[1]
        x = tf.expand_dims(x, 3)
        fillz = tf.zeros_like(x)
        fillz = tf.tile(fillz, [1, 1, 1, strides[1] - 1, 1])
        x = tf.concat((x, fillz), axis=3)
        x = tf.reshape(x, (-1,) + tuple(output_shape))
    if strides[0] > 1:
        output_shape[0] *= strides[0]
        x = tf.expand_dims(x, 2)
        fillz = tf.zeros_like(x)
        fillz = tf.tile(fillz, [1, 1, strides[0] - 1, 1, 1])
        x = tf.concat((x, fillz), axis=2)
        x = tf.reshape(x, (-1,) + tuple(output_shape))
    return x


def _maybe_transpose_kernel(w, transpose=False):
    """Transpose 4-D kernel: permutation of axes 2 and 3 + reverse axes 0 and 1."""
    if not transpose:
        return w
    w_adj = tf.transpose(w, perm=[0, 1, 3, 2])
    w_adj = w_adj[::-1, ::-1, :]
    return w_adj


@tf.function
def process_labels_for_multi_gpu(labels):
    """Process labels to be fed to any loss based on KR estimation with a multi-GPU/TPU
    strategy.

    When using a multi-GPU/TPU strategy, the flag `multi_gpu` in KR-based losses must be
    set to True and the labels have to be pre-processed with this function.

    For binary classification, the labels should be of shape [batch_size, 1].
    For multiclass problems, the labels must be one-hot encoded (1 or 0) with shape
    [batch_size, number of classes].

    Args:
        labels (tf.Tensor): tensor containing the labels

    Returns:
        tf.Tensor: labels processed for KR-based losses with multi-GPU/TPU strategy.
    """
    eps = 1e-7
    labels = tf.cast(tf.where(labels > 0, 1, 0), labels.dtype)
    batch_size = tf.cast(tf.shape(labels)[0], labels.dtype)
    counts = tf.reduce_sum(labels, axis=0)

    pos = labels / (counts + eps)
    neg = (1 - labels) / (batch_size - counts + eps)
    # Since element-wise KR terms are averaged by loss reduction later on, it is needed
    # to multiply by batch_size here.
    return batch_size * (pos - neg)
