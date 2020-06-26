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
