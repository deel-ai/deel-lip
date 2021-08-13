# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains losses used in wasserstein distance estimation. See
https://arxiv.org/abs/2006.06520 for more information.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss
from .utils import _deel_export


@tf.function
@_deel_export
def KR_loss(y_true, y_pred):
    r"""
    Loss to estimate wasserstein-1 distance using Kantorovich-Rubinstein duality.
    The Kantorovich-Rubinstein duality is formulated as following:

    .. math::
        W_1(\mu, \nu) =
        \sup_{f \in Lip_1(\Omega)} \underset{\textbf{x} \sim \mu}{\mathbb{E}}
        \left[f(\textbf{x} )\right] -
        \underset{\textbf{x}  \sim \nu}{\mathbb{E}} \left[f(\textbf{x} )\right]


    Where mu and nu stands for the two distributions, the distribution where the
    label is 1 and the rest.

    Returns:
        Callable, the function to compute Wasserstein loss

    """
    y_true = tf.cast(y_true, y_pred.dtype)
    # create two boolean masks each selecting one distribution
    S0 = tf.equal(y_true, 1)
    S1 = tf.not_equal(y_true, 1)
    # compute the KR dual representation
    return tf.reduce_mean(tf.boolean_mask(y_pred, S0)) - tf.reduce_mean(
        tf.boolean_mask(y_pred, S1)
    )


@tf.function
@_deel_export
def neg_KR_loss(y_true, y_pred):
    r"""
    Loss to compute the negative wasserstein-1 distance using Kantorovich-Rubinstein
    duality. This allows the maximisation of the term using conventional optimizer.

    Returns:
        Callable, the function to compute negative Wasserstein loss

    """
    return -KR_loss(y_true, y_pred)


@_deel_export
class HKR_loss(Loss):
    def __init__(self, alpha, min_margin=1.0, *args, **kwargs):
        r"""
        Wasserstein loss with a regularization param based on hinge loss.

        .. math::
            \inf_{f \in Lip_1(\Omega)} \underset{\textbf{x} \sim P_-}{\mathbb{E}}
            \left[f(\textbf{x} )\right] - \underset{\textbf{x}  \sim P_+}
            {\mathbb{E}} \left[f(\textbf{x} )\right] + \alpha
            \underset{\textbf{x}}{\mathbb{E}} \left(\text{min_margin}
            -Yf(\textbf{x})\right)_+

        Args:
            alpha: regularization factor
            min_margin: minimal margin ( see hinge_margin_loss )
            Kantorovich-rubinstein term of the loss. In order to be consistent
            between hinge and KR, the first label must yield the positve class
            while the second yields negative class.

        Returns:
             a function that compute the regularized Wasserstein loss

        """
        self.alpha = alpha
        self.min_margin = min_margin
        self.KR = KR_loss
        self.hinge = HingeMarginLoss(min_margin)
        super(HKR_loss, self).__init__(*args, **kwargs)

    @tf.function
    def call(self, y_true, y_pred):
        if self.alpha == np.inf:  # if alpha infinite, use hinge only
            return self.hinge(y_true, y_pred)
        else:
            # true value: positive value should be the first to be coherent with the
            # hinge loss (positive y_pred)
            return self.alpha * self.hinge(y_true, y_pred) - self.KR(y_true, y_pred)

    def get_config(self):
        config = {
            "alpha": self.alpha,
            "min_margin": self.min_margin,
        }
        base_config = super(HKR_loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@_deel_export
class HingeMarginLoss(Loss):
    def __init__(self, min_margin=1.0, eps=1e-7, *args, **kwargs):
        r"""
        Compute the hinge margin loss.

        .. math::
            \underset{\textbf{x}}{\mathbb{E}} \left(\text{min_margin}
            -Yf(\textbf{x})\right)_+

        Args:
            min_margin: the minimal margin to enforce.
            eps: small value used to handle both (1,0) and (1,-1) labels. Uses
                sign(y_true-eps) to convert labels.

        Returns:
            a function that compute the hinge loss.

        """
        self.min_margin = min_margin
        self.eps = eps
        super(HingeMarginLoss, self).__init__(*args, **kwargs)

    @tf.function
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        sign = tf.sign(
            y_true - self.eps
        )  # subtracting a small eps makes the loss work for (1,0) and (1,-1) labels
        hinge = tf.nn.relu(self.min_margin - sign * y_pred)
        return tf.reduce_mean(hinge)

    def get_config(self):
        config = {
            "min_margin": self.min_margin,
            "eps": self.eps,
        }
        base_config = super(HingeMarginLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@_deel_export
class KRMulticlassLoss(Loss):
    def __init__(self, eps=1e-7, *args, **kwargs):
        r"""
        Loss to estimate average of W1 distance using Kantorovich-Rubinstein duality over
        outputs. Note y_true should be one hot encoding (labels being 1s and 0s ). In
        this multiclass setup thr KR term is computed for each class and then averaged.

        Args:
            eps: a small positive to avoid zero division when a class is missing. This
            does not impact results as the case leading to a zero denominator also imply
            a zero numerator.

        Returns:
            Callable, the function to compute Wasserstein multiclass loss.
            #Note y_true has to be one hot encoded

        """
        self.eps = eps
        super(KRMulticlassLoss, self).__init__(*args, **kwargs)

    @tf.function
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        # use y_true to zero out y_pred where y_true != 1
        # espYtrue is the avg value of y_pred when y_true==1
        # (one average per output neuron)
        espYtrue = tf.reduce_sum(y_pred * y_true, axis=0) / (
            tf.reduce_sum(y_true, axis=0) + self.eps
        )
        # use(1- y_true) to zero out y_pred where y_true == 1
        # espNotYtrue is the avg value of y_pred when y_true==0
        # (one average per output neuron)
        espNotYtrue = tf.reduce_sum(y_pred * (1 - y_true), axis=0) / (
            (
                tf.cast(tf.shape(y_true)[0], dtype="float32")
                - tf.reduce_sum(y_true, axis=0)
            )
            + self.eps
        )
        # compute the differences to have the KR term for each output neuron,
        # and compute the average over the classes
        return tf.reduce_mean(-espNotYtrue + espYtrue)

    def get_config(self):
        config = {
            "eps": self.eps,
        }
        base_config = super(KRMulticlassLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@_deel_export
class HingeMulticlassLoss(Loss):
    def __init__(self, min_margin=1.0, eps=1e-7, *args, **kwargs):
        """
        Loss to estimate the Hinge loss in a multiclass setup. It compute the elementwise
        hinge term. Note that this formulation differs from the one commonly found in
        tensorflow/pytorch (with marximise the difference between the two largest
        logits). This formulation is consistent with the binary classification loss used
        in a multiclass fashion. Note y_true should be one hot encoded. labels in (1,0)

        Returns:
            Callable, the function to compute multiclass Hinge loss
            #Note y_true has to be one hot encoded

        """
        self.min_margin = min_margin
        self.eps = eps
        super(HingeMulticlassLoss, self).__init__(*args, **kwargs)

    @tf.function
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        sign = tf.sign(y_true - self.eps)
        # subtracting a small eps makes the loss work for (1,0) and (1,-1) labels
        # compute the elementwise hinge term
        hinge = tf.nn.relu(self.min_margin - sign * y_pred)
        # reweight positive elements
        if len(tf.shape(y_pred)) == 2:
            factor = y_true.shape[-1] - 1
        else:
            factor = 1.
        hinge = tf.where(sign > 0, hinge * factor, hinge)
        return tf.reduce_mean(hinge)

    def get_config(self):
        config = {
            "min_margin": self.min_margin,
            "eps": self.eps,
        }
        base_config = super(HingeMulticlassLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@_deel_export
def HKR_multiclass_loss(alpha=0.0, min_margin=1):
    """
    The multiclass version of HKR. This is done by computing the HKR term over each
    class and averaging the results.

    Args:
        alpha: regularization factor
        min_margin: minimal margin ( see Hinge_multiclass_loss )

    Returns:
        Callable, the function to compute HKR loss
        #Note y_true has to be one hot encoded
    """
    hingeloss = HingeMulticlassLoss(min_margin)
    KRloss = KRMulticlassLoss()

    @tf.function
    def HKR_multiclass_loss_fct(y_true, y_pred):
        if alpha == np.inf:  # alpha = inf => hinge only
            return hingeloss(y_true, y_pred)
        elif alpha == 0.0:  # alpha = 0 => KR only
            return -KRloss(y_true, y_pred)
        else:
            a = -KRloss(y_true, y_pred)
            b = hingeloss(y_true, y_pred)
            return a + alpha * b

    return HKR_multiclass_loss_fct


@_deel_export
def MultiMarginLoss(min_margin=1):
    """
    Compute the mean hinge margin loss for multi class (equivalent to Pytorch
     multi_margin_loss)

    Args:
        min_margin: the minimal margin to enforce.
        y_true has to be to_categorical

    Returns:
        Callable, the function to compute multi margin loss
    """

    @tf.function
    def MultiMargin_fct(y_true, y_pred):
        # get the y_pred[target_class]
        # (zeroing out all elements of y_pred where y_true=0)
        vYtrue = tf.reduce_sum(y_pred * y_true, axis=1, keepdims=True)
        # computing elementwise margin term : margin + y_pred[i]-y_pred[target_class]
        margin = tf.nn.relu(min_margin - vYtrue + y_pred)
        # averaging on all outputs and batch
        final_loss = tf.reduce_mean(
            tf.where(y_true == 1, 0.0, margin)
        )  # two steps is useless
        return final_loss

    return MultiMargin_fct
