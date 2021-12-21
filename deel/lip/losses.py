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


@register_keras_serializable("deel-lip", "KR")
class KR(Loss):
    def __init__(self, reduction=Reduction.AUTO, name="KR"):
        r"""
        Loss to estimate Wasserstein-1 distance using Kantorovich-Rubinstein duality.
        The Kantorovich-Rubinstein duality is formulated as following:

        .. math::
            W_1(\mu, \nu) =
            \sup_{f \in Lip_1(\Omega)} \underset{\textbf{x} \sim \mu}{\mathbb{E}}
            \left[f(\textbf{x} )\right] -
            \underset{\textbf{x}  \sim \nu}{\mathbb{E}} \left[f(\textbf{x} )\right]


        Where mu and nu stands for the two distributions, the distribution where the
        label is 1 and the rest.

        Note that `y_true` and `y_pred` must be of rank 2: (batch_size, 1).

        Args:
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        """
        super(KR, self).__init__(reduction=reduction, name=name)

    @tf.function
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        # create two boolean masks each selecting one distribution
        S0 = tf.cast(tf.equal(y_true, 1), y_pred.dtype)
        S1 = tf.cast(tf.not_equal(y_true, 1), y_pred.dtype)
        # compute the KR dual representation
        return tf.reduce_mean(
            y_pred * (S0 / tf.reduce_mean(S0) - S1 / tf.reduce_mean(S1)), axis=-1
        )

    def get_config(self):
        return super(KR, self).get_config()


@register_keras_serializable("deel-lip", "HKR")
class HKR(Loss):
    def __init__(self, alpha, min_margin=1.0, reduction=Reduction.AUTO, name="HKR"):
        r"""
        Wasserstein loss with a regularization parameter based on the hinge margin loss.

        .. math::
            \inf_{f \in Lip_1(\Omega)} \underset{\textbf{x} \sim P_-}{\mathbb{E}}
            \left[f(\textbf{x} )\right] - \underset{\textbf{x}  \sim P_+}
            {\mathbb{E}} \left[f(\textbf{x} )\right] + \alpha
            \underset{\textbf{x}}{\mathbb{E}} \left(\text{min_margin}
            -Yf(\textbf{x})\right)_+

        Note that `y_true` and `y_pred` must be of rank 2: (batch_size, 1).

        Args:
            alpha: regularization factor
            min_margin: minimal margin ( see hinge_margin_loss )
                Kantorovich-Rubinstein term of the loss. In order to be consistent
                between hinge and KR, the first label must yield the positive class
                while the second yields negative class.
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        """
        self.alpha = alpha
        self.min_margin = min_margin
        self.KRloss = KR()
        self.hingeloss = HingeMargin(min_margin)
        super(HKR, self).__init__(reduction=reduction, name=name)

    @tf.function
    def call(self, y_true, y_pred):
        if self.alpha == np.inf:  # alpha = inf => hinge only
            return self.hingeloss.call(y_true, y_pred)
        elif self.alpha == 0.0:  # alpha = 0 => KR only
            return -self.KRloss.call(y_true, y_pred)
        else:
            kr = -self.KRloss.call(y_true, y_pred)
            hinge = self.hingeloss.call(y_true, y_pred)
            return kr + self.alpha * hinge

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

        Note that `y_true` and `y_pred` must be of rank 2: (batch_size, 1).

        Args:
            min_margin: positive float, margin to enforce.
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        """
        self.min_margin = min_margin
        super(HingeMargin, self).__init__(reduction=reduction, name=name)

    @tf.function
    def call(self, y_true, y_pred):
        sign = tf.where(y_true > 0, 1, -1)
        sign = tf.cast(sign, y_pred.dtype)
        hinge = tf.nn.relu(self.min_margin - sign * y_pred)
        return tf.reduce_mean(hinge, axis=-1)

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
        Loss to estimate average of Wasserstein-1 distance using Kantorovich-Rubinstein
        duality over outputs. In this multiclass setup, the KR term is computed for each
        class and then averaged.

        Note that `y_true` and `y_pred` should be one-hot encoded.

        Args:
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        """
        self.eps = 1e-7
        super(MulticlassKR, self).__init__(reduction=reduction, name=name)

    @tf.function
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        batch_size = tf.cast(tf.shape(y_true)[0], dtype="float32")
        num_elements_per_class = tf.reduce_sum(y_true, axis=0)

        pos = y_true / (num_elements_per_class + self.eps)
        neg = (1 - y_true) / (batch_size - num_elements_per_class + self.eps)
        return tf.reduce_mean(batch_size * y_pred * (pos - neg), axis=-1)

    def get_config(self):
        return super(MulticlassKR, self).get_config()


@register_keras_serializable("deel-lip", "MulticlassHinge")
class MulticlassHinge(Loss):
    def __init__(
        self, min_margin=1.0, reduction=Reduction.AUTO, name="MulticlassHinge"
    ):
        """
        Loss to estimate the Hinge loss in a multiclass setup. It computes the
        element-wise Hinge term. Note that this formulation differs from the one
        commonly found in tensorflow/pytorch (which maximises the difference between
        the two largest logits). This formulation is consistent with the binary
        classification loss used in a multiclass fashion.

        Note that `y_true` and `y_pred` should be one-hot encoded.

        Args:
            min_margin: positive float, margin to enforce.
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        """
        self.min_margin = min_margin
        super(MulticlassHinge, self).__init__(reduction=reduction, name=name)

    @tf.function
    def call(self, y_true, y_pred):
        sign = tf.where(y_true > 0, 1, -1)
        sign = tf.cast(sign, y_pred.dtype)
        # compute the elementwise hinge term
        hinge = tf.nn.relu(self.min_margin - sign * y_pred)
        # reweight positive elements
        if (len(tf.shape(y_pred)) == 2) and (tf.shape(y_pred)[-1] != 1):
            factor = y_true.shape[-1] - 1.0
        else:
            factor = 1.0
        hinge = tf.where(sign > 0, hinge * factor, hinge)
        return tf.reduce_mean(hinge, axis=-1)

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

        Note that `y_true` and `y_pred` should be one-hot encoded.

        Args:
            alpha: regularization factor
            min_margin: positive float, margin to enforce.
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        """
        self.alpha = alpha
        self.min_margin = min_margin
        self.hingeloss = MulticlassHinge(self.min_margin)
        self.KRloss = MulticlassKR(name=name)
        super(MulticlassHKR, self).__init__(reduction=reduction, name=name)

    @tf.function
    def call(self, y_true, y_pred):
        if self.alpha == np.inf:  # alpha = inf => hinge only
            return self.hingeloss.call(y_true, y_pred)
        elif self.alpha == 0.0:  # alpha = 0 => KR only
            return -self.KRloss.call(y_true, y_pred)
        else:
            kr = -self.KRloss.call(y_true, y_pred)
            hinge = self.hingeloss.call(y_true, y_pred)
            return kr + self.alpha * hinge

    def get_config(self):
        config = {
            "alpha": self.alpha,
            "min_margin": self.min_margin,
        }
        base_config = super(MulticlassHKR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "MultiMargin")
class MultiMargin(Loss):
    def __init__(self, min_margin=1.0, reduction=Reduction.AUTO, name="MultiMargin"):
        """
        Compute the hinge margin loss for multiclass (equivalent to Pytorch
        multi_margin_loss)

        Note that `y_true` and `y_pred` should be one-hot encoded.

        Args:
            min_margin: positive float, margin to enforce.
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        Notes:
            y_true has to be to_categorical

        """
        self.min_margin = min_margin
        super(MultiMargin, self).__init__(reduction=reduction, name=name)

    @tf.function
    def call(self, y_true, y_pred):
        y_true = tf.where(y_true > 0, 1, 0)
        y_true = tf.cast(y_true, y_pred.dtype)
        # get the y_pred[target_class]
        # (zeroing out all elements of y_pred where y_true=0)
        vYtrue = tf.reduce_sum(y_pred * y_true, axis=-1, keepdims=True)
        # computing elementwise margin term : margin + y_pred[i]-y_pred[target_class]
        margin = tf.nn.relu(self.min_margin - vYtrue + y_pred)
        # averaging on all outputs and batch
        final_loss = tf.reduce_mean(
            tf.where(y_true == 1, 0.0, margin), axis=-1
        )  # two steps is useless
        return final_loss

    def get_config(self):
        config = {
            "min_margin": self.min_margin,
        }
        base_config = super(MultiMargin, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
