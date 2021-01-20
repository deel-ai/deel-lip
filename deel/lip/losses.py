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
def KR_loss(true_values=(1, -1)):
    r"""
    Loss to estimate wasserstein-1 distance using Kantorovich-Rubinstein duality.
    The Kantorovich-Rubinstein duality is formulated as following:

    .. math::
        W_1(\mu, \nu) =
        \sup_{f \in Lip_1(\Omega)} \underset{\textbf{x} \sim \mu}{\mathbb{E}} \left[f(\textbf{x} )\right] -
        \underset{\textbf{x}  \sim \nu}{\mathbb{E}} \left[f(\textbf{x} )\right]

    Args:
        true_values: tuple containing the two label for each predicted class.

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
    r"""
    Loss to compute the negative wasserstein-1 distance using Kantorovich-Rubinstein
    duality. This allows the maximisation of the term using conventional optimizer.

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
    r"""
    Wasserstein loss with a regularization param based on hinge loss.

    .. math::
        \inf_{f \in Lip_1(\Omega)} \underset{\textbf{x} \sim P_-}{\mathbb{E}} \left[f(\textbf{x} )\right] -
        \underset{\textbf{x}  \sim P_+}{\mathbb{E}} \left[f(\textbf{x} )\right] +
        \alpha \underset{\textbf{x}}{\mathbb{E}} \left(\text{min_margin}-Yf(\textbf{x})\right)_+

    Args:
        alpha: regularization factor
        min_margin: minimal margin ( see hinge_margin_loss )
        true_values: tuple containing the two label for each predicted class, used to compute the
        Kantorovich-rubinstein term of the loss

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
    r"""
    Compute the hinge margin loss.

    .. math::
        \underset{\textbf{x}}{\mathbb{E}} \left(\text{min_margin}-Yf(\textbf{x})\right)_+

    Args:
        min_margin: the minimal margin to enforce.

    Returns:
        a function that compute the hinge loss.

    """

    @tf.function
    def hinge_margin_fct(y_true, y_pred):
        sign = K.sign(y_true)
        hinge = K.maximum(0.0, min_margin - sign * y_pred)
        return K.mean(hinge)

    return hinge_margin_fct


@_deel_export
def KR_multiclass_loss():
    r"""
    Loss to estimate average of W1 distance using Kantorovich-Rubinstein duality over outputs.
    Note y_true should be one hot encoding (labels being 1s and 0s ).

    Returns:
        Callable, the function to compute Wasserstein multiclass loss

    """

    @tf.function
    def KR_multiclass_loss_fct(y_true, y_pred):
        espNotYtrue = tf.reduce_sum(y_pred * (1 - y_true), axis=0) / (
            tf.cast(tf.shape(y_true)[0], dtype="float32")
            - tf.reduce_sum(y_true, axis=0)
        )
        espYtrue = tf.reduce_sum(y_pred * y_true, axis=0) / tf.reduce_sum(
            y_true, axis=0
        )
        return tf.reduce_mean(-espNotYtrue + espYtrue)

    return KR_multiclass_loss_fct


@_deel_export
def Hinge_multiclass_loss(min_margin=1):
    """
    Loss to estimate the Hinge loss in a multiclass setup. It compute
    Note y_true should be one hot encoding.

    Returns:
        Callable, the function to compute Hinge loss

    """

    @tf.function
    def Hinge_multiclass_loss_fct(y_true, y_pred):
        sign = 2 * y_true - 1
        hinge = tf.nn.relu(min_margin - sign * y_pred)
        return K.mean(hinge)

    return Hinge_multiclass_loss_fct


@_deel_export
def HKR_multiclass_loss(alpha=0.0, min_margin=1):
    """
    Loss to estimate the Hinge loss for each output.

    Args:
        #Note y_true should be one hot encoding

    Returns:
        Callable, the function to compute Hinge loss

    """
    hingeloss = Hinge_multiclass_loss(min_margin)
    KRloss = KR_multiclass_loss()
    # print("Warning 1/alpha KRLoss => to get the same level of hinge in MultiMargin")

    @tf.function
    def HKR_multiclass_loss_fct(y_true, y_pred):
        if alpha == np.inf:  # alpha negative hinge only
            return hingeloss(y_true, y_pred)
        elif alpha == 0.0:
            return -KRloss(y_true, y_pred)
        else:
            return -KRloss(y_true, y_pred) / alpha + hingeloss(y_true, y_pred)

    return HKR_multiclass_loss_fct


'''@_deel_export
def MultiMarginLoss(min_margin=1):
    """
    Compute the hinge margin loss for multi class
    Args:
        min_margin: the minimal margin to enforce.
        y_true has to be to_categorical
    Returns:

    """

    @tf.function
    def MultiMargin_fct(y_true, y_pred):
        H1 = tf.where(y_true==1,tf.reduce_min(y_pred), y_pred) 
        H = tf.reduce_max(H1, 1,keepdims=True)    
        L = tf.nn.relu((min_margin - y_pred + H) * y_true)
        final_loss = tf.reduce_mean(tf.reduce_max(L, 1))
        return final_loss

    return MultiMargin_fct
'''


@_deel_export
def MultiMarginLoss(min_margin=1):
    """
    Compute the mean hinge margin loss for multi class (equivalent to Pytorch multi_margin_loss)
    Args:
        min_margin: the minimal margin to enforce.
        y_true has to be to_categorical
    Returns:

    """

    @tf.function
    def MultiMargin_fct(y_true, y_pred):
        vYtrue = tf.reduce_sum(y_pred * y_true, axis=1, keepdims=True)
        margin = tf.nn.relu(min_margin - vYtrue + y_pred)
        final_loss = tf.reduce_mean(
            tf.where(y_true == 1, 0.0, margin)
        )  ## two steps is useless
        return final_loss

    return MultiMargin_fct
