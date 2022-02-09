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
from tensorflow.keras import backend as K


def evaluate_lip_const_gen(
    model: Model,
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
        eps: magnitude of noise to add to input in order to compute the constant
        seed: seed used when generating the noise ( can be set to None )

    Returns:
        the empirically evaluated lipschitz constant.

    """
    x, y = generator.send(None)
    return evaluate_lip_const(model, x, eps, seed=seed)


def evaluate_lip_const(model: Model, x, eps=1e-4, seed=None):
    """
    Evaluate the Lipschitz constant of a model, with the naive method.
    Please note that the estimation of the lipschitz constant is done locally around
    input sample. This may not correctly estimate the behaviour in the whole domain.

    Args:
        model: built keras model used to make predictions
        x: inputs used to compute the lipschitz constant
        eps: magnitude of noise to add to input in order to compute the constant
        seed: seed used when generating the noise ( can be set to None )

    Returns:
        the empirically evaluated lipschitz constant. The computation might also be
        inaccurate in high dimensional space.

    """
    y_pred = model.predict(x)
    # x = np.repeat(x, 100, 0)
    # y_pred = np.repeat(y_pred, 100, 0)
    x_var = x + K.random_uniform(
        shape=x.shape, minval=eps * 0.25, maxval=eps, seed=seed
    )
    y_pred_var = model.predict(x_var)
    dx = x - x_var
    dfx = y_pred - y_pred_var
    ndx = K.sqrt(K.sum(K.square(dx), axis=range(1, len(x.shape))))
    ndfx = K.sqrt(K.sum(K.square(dfx), axis=range(1, len(y_pred.shape))))
    lip_cst = K.max(ndfx / ndx)
    print("lip cst: %.3f" % lip_cst)
    return lip_cst


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
        labels: tf.Tensor containing the labels
    Returns:
        labels processed for KR-based losses with multi-GPU/TPU strategy.
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
