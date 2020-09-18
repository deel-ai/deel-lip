# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains losses used in wasserstein distance estimation.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from .utils import _deel_export


@_deel_export
def KR_loss(true_values=(0, 1)):
    """
    Loss to estimate wasserstein-1 distance using Kantorovich-Rubinstein duality.

    Args:
        true_values: tuple containing the two label for each predicted class

    Returns:
        Callable, the function to compute Wasserstein loss

    """

    @tf.function
    def KR_loss_fct(y_true, y_pred):
        S0 = tf.equal(y_true, true_values[0])
        S1 = tf.equal(y_true, true_values[1])
        return K.mean(tf.boolean_mask(y_pred, S0)) - K.mean(tf.boolean_mask(y_pred, S1))

    return KR_loss_fct


@_deel_export
def neg_KR_loss(true_values=(1, -1)):
    """
    Loss to compute the negative wasserstein-1 distance using Kantorovich-Rubinstein
    duality.

    Args:
        true_values: tuple containing the two label for each predicted class

    Returns:
        Callable, the function to compute negative Wasserstein loss

    """

    @tf.function
    def neg_KR_loss_fct(y_true, y_pred):
        return -KR_loss(true_values)(y_true, y_pred)

    return neg_KR_loss_fct


@_deel_export
def HKR_loss(alpha, min_margin=1, true_values=(1, -1)):
    """
    Wasserstein loss with a regularization param based on hinge loss.

    Args:
        alpha: regularization factor
        min_margin: minimal margin ( see hinge_margin_loss )
        true_values: tuple containing the two label for each predicted class

    Returns:
         a function that compute the regularized Wasserstein loss

    """

    @tf.function
    def HKR_loss_fct(y_true, y_pred):
        if alpha == np.inf:  # alpha negative hinge only
            return hinge_margin_loss(min_margin)(y_true, y_pred)
        else:
            # true value: positive value should be the first to be coherent with the
            # hinge loss (positive y_pred)
            return alpha * hinge_margin_loss(min_margin)(y_true, y_pred) - KR_loss(
                true_values
            )(y_true, y_pred)

    return HKR_loss_fct


@_deel_export
def hinge_margin_loss(min_margin=1):
    """
    Compute the hinge margin loss.

    Args:
        min_margin: the minimal margin to enforce.

    Returns:
        a function that compute the hinge loss

    """

    @tf.function
    def hinge_margin_fct(y_true, y_pred):
        sign = K.sign(y_true)
        hinge = K.maximum(0.0, min_margin - sign * y_pred)
        return K.mean(hinge)

    return hinge_margin_fct


@_deel_export
class HKR_multiclass():
    """
    Multiclass version of the HKR loss one vs. all version.
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.__name__ = "HKR_multiclass"

    @tf.function
    def __call__(self, y_true, y_pred):
        pos_values = tf.where(y_true == 1, y_pred, 0.0)
        neg_values = tf.where(y_true != 1, y_pred, 0.0)
        pos_min = tf.reduce_min(pos_values, axis=-1)
        neg_max = tf.reduce_max(neg_values, axis=-1)
        return -(
            self.alpha * tf.reduce_mean(pos_min - neg_max, axis=0)
            + tf.reduce_min(
            (tf.reduce_sum(pos_values, axis=0)/tf.reduce_sum(tf.cast(y_true == 1, pos_values.dtype), axis=0)) -
            (tf.reduce_sum(neg_values, axis=0)/tf.reduce_sum(tf.cast(y_true != 1, neg_values.dtype), axis=0)), axis=-1)
        )

    def get_config(self):
        config = {"alpha": self.alpha}
        return config


@_deel_export
def one_versus_all_HKR(alpha, min_margin=1, true_values=(1, -1)):
    """
    Wasserstein loss with a regularization param based on hinge loss for the multilabel case.

    Args:
        alpha: regularization factor
        min_margin: minimal margin ( see hinge_margin_loss )
        true_values: tuple containing the two label for each predicted class,
                     one must be positive and the other negative

    Returns:
         a function that compute the regularized Wasserstein loss.
         In the following B is the batch size, and K the number of classes.
            y_true: tensor of shape (B, K) containing y_true[0] or y_true[1]
            y_pred: tesor of shape (B, K) containing logits

    Remark: this function works even when many y_true[0] are present on the same line
            in this case it corresponds to the multilabel case, where an example can belong to
            multiple classes simultaneously.
    """
    @tf.function
    def one_versus_all_HKR_loss_fct(y_true, y_pred):
        sign = tf.cast(tf.sign(y_true), dtype=tf.float32)
        margin_dist = min_margin - sign * y_pred  # shape (B, K)
        margin_dist = tf.maximum(margin_dist, 0.0)  # shape (B, K)
        hinge_loss = tf.reduce_mean(margin_dist)  # scalar, average over batch
        if alpha == np.inf:
            return hinge_loss
        one_mask = tf.cast(tf.equal(y_true, true_values[0]), dtype=tf.float32)  # shape (B,K), 1. for true class, 0. otherwise
        all_mask = tf.cast(tf.equal(y_true, true_values[1]), dtype=tf.float32)  # shape (B,K), 1. for other classes, 0. otherwise
        one_avg = tf.reduce_sum(one_mask * y_pred) / tf.reduce_sum(one_mask)  # shape B
        all_avg = tf.reduce_sum(all_mask * y_pred) / tf.reduce_sum(all_mask)  # shape B
        kr_loss = tf.reduce_mean(one_avg - all_avg)  # average over batch
        return alpha * hinge_loss - kr_loss
    return one_versus_all_HKR_loss_fct
