# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains losses used in Wasserstein distance estimation. See
[this paper](https://arxiv.org/abs/2006.06520) for more information.
"""
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy, Loss, Reduction
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable("deel-lip", "_kr")
def _kr(y_true, y_pred, epsilon):
    """Returns the element-wise KR loss.

    `y_true` and `y_pred` must be of rank 2: (batch_size, 1) for binary classification
    or (batch_size, C) for multilabel/multiclass classification (with C categories).
    `y_true` labels should be either 1 and 0, or 1 and -1.
    """
    y_true = tf.cast(y_true, y_pred.dtype)
    batch_size = tf.cast(tf.shape(y_true)[0], dtype=y_pred.dtype)
    # Transform y_true into {1, 0} values
    S1 = tf.cast(tf.equal(y_true, 1), y_pred.dtype)
    num_elements_per_class = tf.reduce_sum(S1, axis=0)

    pos = S1 / (num_elements_per_class + epsilon)
    neg = (1 - S1) / (batch_size - num_elements_per_class + epsilon)
    # Since element-wise KR terms are averaged by loss reduction later on, it is needed
    # to multiply by batch_size here.
    # In binary case (`y_true` of shape (batch_size, 1)), `tf.reduce_mean(axis=-1)`
    # behaves like `tf.squeeze()` to return element-wise loss of shape (batch_size, ).
    return tf.reduce_mean(batch_size * y_pred * (pos - neg), axis=-1)


@register_keras_serializable("deel-lip", "_kr_multi_gpu")
def _kr_multi_gpu(y_true, y_pred):
    """Returns the element-wise KR loss when computing with a multi-GPU/TPU strategy.

    `y_true` and `y_pred` can be either of shape (batch_size, 1) or
    (batch_size, # classes).

    When using this loss function, the labels `y_true` must be pre-processed with the
    `process_labels_for_multi_gpu()` function.
    """
    y_true = tf.cast(y_true, y_pred.dtype)
    # Since the information of batch size was included in `y_true` by
    # `process_labels_for_multi_gpu()`, there is no need here to multiply by batch size.
    # In binary case (`y_true` of shape (batch_size, 1)), `tf.reduce_mean(axis=-1)`
    # behaves like `tf.squeeze()` to return element-wise loss of shape (batch_size, ).
    return tf.reduce_mean(y_pred * y_true, axis=-1)


@register_keras_serializable("deel-lip", "KR")
class KR(Loss):
    def __init__(self, multi_gpu=False, reduction=Reduction.AUTO, name="KR"):
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

        Note that `y_true` and `y_pred` must be of rank 2: (batch_size, 1) or
        (batch_size, C) for multilabel classification (with C categories).
        `y_true` accepts label values in (0, 1), (-1, 1), or pre-processed with the
        `deel.lip.utils.process_labels_for_multi_gpu()` function.

        Using a multi-GPU/TPU strategy requires to set `multi_gpu` to True and to
        pre-process the labels `y_true` with the
        `deel.lip.utils.process_labels_for_multi_gpu()` function.

        Args:
            multi_gpu (bool): set to True when running on multi-GPU/TPU
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        """
        self.eps = 1e-7
        self.multi_gpu = multi_gpu
        super(KR, self).__init__(reduction=reduction, name=name)
        if multi_gpu:
            self.kr_function = _kr_multi_gpu
        else:
            self.kr_function = partial(_kr, epsilon=self.eps)

    @tf.function
    def call(self, y_true, y_pred):
        return self.kr_function(y_true, y_pred)

    def get_config(self):
        config = {"multi_gpu": self.multi_gpu}
        base_config = super(KR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "HKR")
class HKR(Loss):
    def __init__(
        self,
        alpha,
        min_margin=1.0,
        multi_gpu=False,
        reduction=Reduction.AUTO,
        name="HKR",
    ):
        r"""
        Wasserstein loss with a regularization parameter based on the hinge margin loss.

        .. math::
            \inf_{f \in Lip_1(\Omega)} \underset{\textbf{x} \sim P_-}{\mathbb{E}}
            \left[f(\textbf{x} )\right] - \underset{\textbf{x}  \sim P_+}
            {\mathbb{E}} \left[f(\textbf{x} )\right] + \alpha
            \underset{\textbf{x}}{\mathbb{E}} \left(\text{min_margin}
            -Yf(\textbf{x})\right)_+

        Note that `y_true` and `y_pred` must be of rank 2: (batch_size, 1) or
        (batch_size, C) for multilabel classification (with C categories).
        `y_true` accepts label values in (0, 1), (-1, 1), or pre-processed with the
        `deel.lip.utils.process_labels_for_multi_gpu()` function.

        Using a multi-GPU/TPU strategy requires to set `multi_gpu` to True and to
        pre-process the labels `y_true` with the
        `deel.lip.utils.process_labels_for_multi_gpu()` function.

        Args:
            alpha: regularization factor
            min_margin: minimal margin ( see hinge_margin_loss )
                Kantorovich-Rubinstein term of the loss. In order to be consistent
                between hinge and KR, the first label must yield the positive class
                while the second yields negative class.
            multi_gpu (bool): set to True when running on multi-GPU/TPU
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        """
        self.alpha = tf.Variable(alpha, dtype=tf.float32)
        self.min_margin = tf.Variable(min_margin, dtype=tf.float32)
        self.multi_gpu = multi_gpu
        self.KRloss = KR(multi_gpu=multi_gpu)
        if alpha == np.inf:  # alpha = inf => hinge only
            self.fct = partial(hinge_margin, min_margin=self.min_margin)
        else:
            self.fct = self.hkr
        super(HKR, self).__init__(reduction=reduction, name=name)

    @tf.function
    def hkr(self, y_true, y_pred):
        a = -self.KRloss.call(y_true, y_pred)
        b = hinge_margin(y_true, y_pred, self.min_margin)
        return a + self.alpha * b

    def call(self, y_true, y_pred):
        return self.fct(y_true, y_pred)

    def get_config(self):
        config = {
            "alpha": self.alpha.numpy(),
            "min_margin": self.min_margin.numpy(),
            "multi_gpu": self.multi_gpu,
        }
        base_config = super(HKR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def hinge_margin(y_true, y_pred, min_margin):
    """Compute the element-wise binary hinge margin loss.

    Note that `y_true` and `y_pred` must be of rank 2: (batch_size, 1) or
    (batch_size, C) for multilabel classification (with C categories).
    `y_true` accepts label values in (0, 1), (-1, 1), or pre-processed with the
    `deel.lip.utils.process_labels_for_multi_gpu()` function.

    Args:
        min_margin: positive float, margin to enforce.

    Returns:
        Element-wise hinge margin loss value.

    """
    sign = tf.where(y_true > 0, 1, -1)
    sign = tf.cast(sign, y_pred.dtype)
    hinge = tf.nn.relu(min_margin / 2.0 - sign * y_pred)
    # In binary case (`y_true` of shape (batch_size, 1)), `tf.reduce_mean(axis=-1)`
    # behaves like `tf.squeeze` to return element-wise loss of shape (batch_size, ).
    return tf.reduce_mean(hinge, axis=-1)


@register_keras_serializable("deel-lip", "HingeMargin")
class HingeMargin(Loss):
    def __init__(self, min_margin=1.0, reduction=Reduction.AUTO, name="HingeMargin"):
        r"""
        Compute the hinge margin loss.

        .. math::
            \underset{\textbf{x}}{\mathbb{E}} \left(\text{min_margin}
            -Yf(\textbf{x})\right)_+

        Note that `y_true` and `y_pred` must be of rank 2: (batch_size, 1) or
        (batch_size, C) for multilabel classification (with C categories).
        `y_true` accepts label values in (0, 1), (-1, 1), or pre-processed with the
        `deel.lip.utils.process_labels_for_multi_gpu()` function.

        Args:
            min_margin: positive float, margin to enforce.
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        """
        self.min_margin = tf.Variable(min_margin, dtype=tf.float32)
        super(HingeMargin, self).__init__(reduction=reduction, name=name)

    @tf.function
    def call(self, y_true, y_pred):
        return hinge_margin(y_true, y_pred, self.min_margin)

    def get_config(self):
        config = {
            "min_margin": self.min_margin.numpy(),
        }
        base_config = super(HingeMargin, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "MulticlassKR")
class MulticlassKR(Loss):
    def __init__(self, multi_gpu=False, reduction=Reduction.AUTO, name="MulticlassKR"):
        r"""
        Loss to estimate average of Wasserstein-1 distance using Kantorovich-Rubinstein
        duality over outputs. In this multiclass setup, the KR term is computed for each
        class and then averaged.

        Note that `y_true` should be one-hot encoded or pre-processed with the
        `deel.lip.utils.process_labels_for_multi_gpu()` function.

        Using a multi-GPU/TPU strategy requires to set `multi_gpu` to True and to
        pre-process the labels `y_true` with the
        `deel.lip.utils.process_labels_for_multi_gpu()` function.

        Args:
            multi_gpu (bool): set to True when running on multi-GPU/TPU
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        """
        self.eps = 1e-7
        self.multi_gpu = multi_gpu
        super(MulticlassKR, self).__init__(reduction=reduction, name=name)
        if multi_gpu:
            self.kr_function = _kr_multi_gpu
        else:
            self.kr_function = partial(_kr, epsilon=self.eps)

    @tf.function
    def call(self, y_true, y_pred):
        return self.kr_function(y_true, y_pred)

    def get_config(self):
        config = {"multi_gpu": self.multi_gpu}
        base_config = super(MulticlassKR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def multiclass_hinge(y_true, y_pred, min_margin):
    """Compute the multi-class hinge margin loss.

    `y_true` and `y_pred` must be of shape (batch_size, # classes).
    Note that `y_true` should be one-hot encoded or pre-processed with the
    `deel.lip.utils.process_labels_for_multi_gpu()` function.

    Args:
        y_true: tensor of true targets of shape (batch_size, # classes)
        y_pred: tensor of predicted targets of shape (batch_size, # classes)
        min_margin: positive float, margin to enforce.

    Returns:
        Element-wise multi-class hinge margin loss value.
    """
    sign = tf.where(y_true > 0, 1, -1)
    sign = tf.cast(sign, y_pred.dtype)
    # compute the elementwise hinge term
    hinge = tf.nn.relu(min_margin / 2.0 - sign * y_pred)
    # reweight positive elements
    factor = y_pred.shape[-1] - 1.0
    hinge = tf.where(sign > 0, hinge * factor, hinge)
    return tf.reduce_mean(hinge, axis=-1)


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

        Note that `y_true` should be one-hot encoded or pre-processed with the
        `deel.lip.utils.process_labels_for_multi_gpu()` function.

        Args:
            min_margin: positive float, margin to enforce.
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        """
        self.min_margin = tf.Variable(min_margin, dtype=tf.float32)
        super(MulticlassHinge, self).__init__(reduction=reduction, name=name)

    @tf.function
    def call(self, y_true, y_pred):
        return multiclass_hinge(y_true, y_pred, self.min_margin)

    def get_config(self):
        config = {
            "min_margin": self.min_margin.numpy(),
        }
        base_config = super(MulticlassHinge, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "MulticlassHKR")
class MulticlassHKR(Loss):
    def __init__(
        self,
        alpha=10.0,
        min_margin=1.0,
        multi_gpu=False,
        reduction=Reduction.AUTO,
        name="MulticlassHKR",
    ):
        """
        The multiclass version of HKR. This is done by computing the HKR term over each
        class and averaging the results.

        Note that `y_true` should be one-hot encoded or pre-processed with the
        `deel.lip.utils.process_labels_for_multi_gpu()` function.

        Using a multi-GPU/TPU strategy requires to set `multi_gpu` to True and to
        pre-process the labels `y_true` with the
        `deel.lip.utils.process_labels_for_multi_gpu()` function.

        Args:
            alpha: regularization factor
            min_margin: positive float, margin to enforce.
            multi_gpu (bool): set to True when running on multi-GPU/TPU
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        """
        self.alpha = tf.Variable(alpha, dtype=tf.float32)
        self.min_margin = tf.Variable(min_margin, dtype=tf.float32)
        self.multi_gpu = multi_gpu
        self.KRloss = MulticlassKR(multi_gpu=multi_gpu, reduction=reduction, name=name)
        if alpha == np.inf:  # alpha = inf => hinge only
            self.fct = partial(multiclass_hinge, min_margin=self.min_margin)
        else:
            self.fct = self.hkr
        super(MulticlassHKR, self).__init__(reduction=reduction, name=name)

    @tf.function
    def hkr(self, y_true, y_pred):
        a = -self.KRloss.call(y_true, y_pred)
        b = multiclass_hinge(y_true, y_pred, self.min_margin)
        return a + self.alpha * b

    def call(self, y_true, y_pred):
        return self.fct(y_true, y_pred)

    def get_config(self):
        config = {
            "alpha": self.alpha.numpy(),
            "min_margin": self.min_margin.numpy(),
            "multi_gpu": self.multi_gpu,
        }
        base_config = super(MulticlassHKR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "MultiMargin")
class MultiMargin(Loss):
    def __init__(self, min_margin=1.0, reduction=Reduction.AUTO, name="MultiMargin"):
        """
        Compute the hinge margin loss for multiclass (equivalent to Pytorch
        multi_margin_loss)

        Note that `y_true` should be one-hot encoded or pre-processed with the
        `deel.lip.utils.process_labels_for_multi_gpu()` function.

        Args:
            min_margin: positive float, margin to enforce.
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        """
        self.min_margin = tf.Variable(min_margin, dtype=tf.float32)
        super(MultiMargin, self).__init__(reduction=reduction, name=name)

    @tf.function
    def call(self, y_true, y_pred):
        mask = tf.where(y_true > 0, 1, 0)
        mask = tf.cast(mask, y_pred.dtype)
        # get the y_pred[target_class]
        # (zeroing out all elements of y_pred where y_true=0)
        vYtrue = tf.reduce_sum(y_pred * mask, axis=-1, keepdims=True)
        # computing elementwise margin term : margin + y_pred[i]-y_pred[target_class]
        margin = tf.nn.relu(self.min_margin - vYtrue + y_pred)
        # averaging on all outputs and batch
        final_loss = tf.reduce_mean((1.0 - mask) * margin, axis=-1)
        return final_loss

    def get_config(self):
        config = {
            "min_margin": self.min_margin.numpy(),
        }
        base_config = super(MultiMargin, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "CategoricalHinge")
class CategoricalHinge(Loss):
    def __init__(self, min_margin, reduction=Reduction.AUTO, name="CategoricalHinge"):
        """
        Similar to original categorical hinge, but with a settable margin parameter.
        This implementation is sligthly different from the Keras one.

        `y_true` and `y_pred` must be of shape (batch_size, # classes).
        Note that `y_true` should be one-hot encoded or pre-processed with the
        `deel.lip.utils.process_labels_for_multi_gpu()` function.

        Args:
            min_margin: positive float, margin parameter.
            reduction: reduction of the loss, passed to original loss.
            name: name of the loss
        """
        self.min_margin = tf.Variable(min_margin, dtype=tf.float32)
        super(CategoricalHinge, self).__init__(name=name, reduction=reduction)

    def call(self, y_true, y_pred):
        mask = tf.where(y_true > 0, 1, 0)
        mask = tf.cast(mask, y_pred.dtype)
        pos = tf.reduce_sum(mask * y_pred, axis=-1)
        neg = tf.reduce_max(tf.where(mask > 0, tf.float32.min, y_pred), axis=-1)
        return tf.nn.relu(self.min_margin - (pos - neg))

    def get_config(self):
        config = {"min_margin": self.min_margin.numpy()}
        base_config = super(CategoricalHinge, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "TauCategoricalCrossentropy")
class TauCategoricalCrossentropy(Loss):
    def __init__(
        self, tau, reduction=Reduction.AUTO, name="TauCategoricalCrossentropy"
    ):
        """
        Similar to original categorical crossentropy, but with a settable temperature
        parameter.

        Args:
            tau: temperature parameter.
            reduction: reduction of the loss, passed to original loss.
            name: name of the loss
        """
        self.tau = tf.Variable(tau, dtype=tf.float32)
        super(TauCategoricalCrossentropy, self).__init__(name=name, reduction=reduction)

    def call(self, y_true, y_pred, *args, **kwargs):
        return (
            categorical_crossentropy(
                y_true, self.tau * y_pred, from_logits=True, *args, **kwargs
            )
            / self.tau
        )

    def get_config(self):
        config = {"tau": self.tau.numpy()}
        base_config = super(TauCategoricalCrossentropy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
