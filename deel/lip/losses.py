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
from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import Reduction
from tensorflow.keras.utils import register_keras_serializable


@tf.function
@register_keras_serializable("deel-lip", "KR")
def KR(y_true, y_pred):
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
@register_keras_serializable("deel-lip", "negative_KR")
def negative_KR(y_true, y_pred):
    r"""
    Loss to compute the negative wasserstein-1 distance using Kantorovich-Rubinstein
    duality. This allows the maximisation of the term using conventional optimizer.

    Returns:
        Callable, the function to compute negative Wasserstein loss

    """
    return -KR(y_true, y_pred)


@register_keras_serializable("deel-lip", "HKR")
class HKR(Loss):
    def __init__(self, alpha, min_margin=1.0, reduction=Reduction.AUTO, name="HKR"):
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
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        Returns:
             a function that compute the regularized Wasserstein loss

        """
        self.alpha = alpha
        self.min_margin = min_margin
        self.KR = KR
        self.hinge = HingeMargin(min_margin)
        super(HKR, self).__init__(reduction=reduction, name=name)

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
        base_config = super(HKR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "HingeMargin")
class HingeMargin(Loss):
    def __init__(self, min_margin=1.0, reduction=Reduction.AUTO, name="HingeMargin"):
        r"""
        Compute the hinge margin loss.

        .. math::
            \underset{\textbf{x}}{\mathbb{E}} \left(\text{min_margin}
            -Yf(\textbf{x})\right)_+

        Args:
            min_margin: the minimal margin to enforce.
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        Returns:
            a function that compute the hinge loss.

        """
        self.min_margin = min_margin
        super(HingeMargin, self).__init__(reduction=reduction, name=name)

    @tf.function
    def call(self, y_true, y_pred):
        y_true = tf.where(y_true == 1, 1, -1)
        sign = tf.cast(y_true, y_pred.dtype)
        hinge = tf.nn.relu(self.min_margin - sign * y_pred)
        return tf.reduce_mean(hinge)

    def get_config(self):
        config = {
            "min_margin": self.min_margin,
        }
        base_config = super(HingeMargin, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "MulticlassKR")
class MulticlassKR(Loss):
    def __init__(self, reduction=Reduction.AUTO, name="MulticlassKR"):
        r"""
        Loss to estimate average of W1 distance using Kantorovich-Rubinstein duality
        over outputs. Note y_true should be one hot encoding (labels being 1s and 0s
        ). In this multiclass setup thr KR term is computed for each class and then
        averaged.

        Args:
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        Returns:
            Callable, the function to compute Wasserstein multiclass loss.
            #Note y_true has to be one hot encoded

        """
        self.eps = 1e-7
        super(MulticlassKR, self).__init__(reduction=reduction, name=name)

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
        return super(MulticlassKR, self).get_config()


@register_keras_serializable("deel-lip", "MulticlassHinge")
class MulticlassHinge(Loss):
    def __init__(
        self, min_margin=1.0, reduction=Reduction.AUTO, name="MulticlassHinge"
    ):
        """
        Loss to estimate the Hinge loss in a multiclass setup. It compute the
        elementwise hinge term. Note that this formulation differs from the one
        commonly found in tensorflow/pytorch (with marximise the difference between
        the two largest logits). This formulation is consistent with the binary
        classification loss used in a multiclass fashion. Note y_true should be one
        hot encoded. labels in (1,0)

        Args:
            min_margin: positive float, margin to enforce.
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        Returns:
            Callable, the function to compute multiclass Hinge loss
            #Note y_true has to be one hot encoded

        """
        self.min_margin = min_margin
        super(MulticlassHinge, self).__init__(reduction=reduction, name=name)

    @tf.function
    def call(self, y_true, y_pred):
        sign = tf.where(y_true == 1, 1.0, -1.0)
        y_true = tf.cast(y_true, y_pred.dtype)
        # compute the elementwise hinge term
        hinge = tf.nn.relu(self.min_margin - sign * y_pred)
        # reweight positive elements
        if (len(tf.shape(y_pred)) == 2) and (tf.shape(y_pred)[-1] != 1):
            factor = y_true.shape[-1] - 1.0
        else:
            factor = 1.0
        hinge = tf.where(sign > 0, hinge * factor, hinge)
        return tf.reduce_mean(hinge)

    def get_config(self):
        config = {
            "min_margin": self.min_margin,
        }
        base_config = super(MulticlassHinge, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "MulticlassHKR")
class MulticlassHKR(Loss):
    def __init__(
        self,
        alpha=10.0,
        min_margin=1.0,
        reduction=Reduction.AUTO,
        name="MulticlassHKR",
    ):
        """
        The multiclass version of HKR. This is done by computing the HKR term over each
        class and averaging the results.

        Args:
            alpha: regularization factor
            min_margin: minimal margin ( see Hinge_multiclass_loss )
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        Returns:
            Callable, the function to compute HKR loss
            #Note y_true has to be one hot encoded
        """
        self.alpha = alpha
        self.min_margin = min_margin
        self.hingeloss = MulticlassHinge(self.min_margin)
        self.KRloss = MulticlassKR(reduction=reduction, name=name)
        super(MulticlassHKR, self).__init__(reduction=reduction, name=name)

    @tf.function
    def call(self, y_true, y_pred):
        if self.alpha == np.inf:  # alpha = inf => hinge only
            return self.hingeloss(y_true, y_pred)
        elif self.alpha == 0.0:  # alpha = 0 => KR only
            return -self.KRloss(y_true, y_pred)
        else:
            a = -self.KRloss(y_true, y_pred)
            b = self.hingeloss(y_true, y_pred)
            return a + self.alpha * b

    def get_config(self):
        config = {
            "alpha": self.alpha,
            "min_margin": self.min_margin,
        }
        base_config = super(MulticlassHKR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "MultiMargin")
class MultiMargin(Loss):
    def __init__(self, min_margin=1, reduction=Reduction.AUTO, name="MultiMargin"):
        """
        Compute the mean hinge margin loss for multi class (equivalent to Pytorch
         multi_margin_loss)

        Args:
            min_margin: the minimal margin to enforce.
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        Notes:
            y_true has to be to_categorical

        Returns:
            Callable, the function to compute multi margin loss
        """
        self.min_margin = min_margin
        super(MultiMargin, self).__init__(reduction=reduction, name=name)

    @tf.function
    def call(self, y_true, y_pred):
        y_true = tf.where(y_true == 1, 1, 0)
        y_true = tf.cast(y_true, y_pred.dtype)
        # get the y_pred[target_class]
        # (zeroing out all elements of y_pred where y_true=0)
        vYtrue = tf.reduce_sum(y_pred * y_true, axis=-1, keepdims=True)
        # computing elementwise margin term : margin + y_pred[i]-y_pred[target_class]
        margin = tf.nn.relu(self.min_margin - vYtrue + y_pred)
        # averaging on all outputs and batch
        final_loss = tf.reduce_mean(
            tf.where(y_true == 1, 0.0, margin)
        )  # two steps is useless
        return final_loss

    def get_config(self):
        config = {
            "min_margin": self.min_margin,
        }
        base_config = super(MultiMargin, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
