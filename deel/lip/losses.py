# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains losses used in wasserstein distance estimation. See
https://arxiv.org/abs/2006.06520 for more information.
"""
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import Reduction
from tensorflow.keras.losses import categorical_crossentropy
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
    :func:`process_labels_for_multi_gpu()` function.
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
        :func:`deel.lip.utils.process_labels_for_multi_gpu()` function.

        Using a multi-GPU/TPU strategy requires to set `multi_gpu` to True and to
        pre-process the labels `y_true` with the
        :func:`deel.lip.utils.process_labels_for_multi_gpu()` function.

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
        :func:`deel.lip.utils.process_labels_for_multi_gpu()` function.

        Using a multi-GPU/TPU strategy requires to set `multi_gpu` to True and to
        pre-process the labels `y_true` with the
        :func:`deel.lip.utils.process_labels_for_multi_gpu()` function.

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
        self.min_margin = min_margin
        self.multi_gpu = multi_gpu
        self.KRloss = KR(multi_gpu=multi_gpu)
        if not hasattr(self, "hingeloss"):  # compatibility with HKRauto
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
            "alpha": self.alpha.numpy(),
            "min_margin": self.min_margin,
            "multi_gpu": self.multi_gpu,
        }
        base_config = super(HKR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "HKRauto")
class HKRauto(HKR):
    def __init__(
        self,
        alpha,
        min_margin=1.0,
        max_margin=200,
        alpha_margin=0.1,
        multi_gpu=False,
        reduction=Reduction.AUTO,
        name="HKRauto",
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
        :func:`deel.lip.utils.process_labels_for_multi_gpu()` function.

        Using a multi-GPU/TPU strategy requires to set `multi_gpu` to True and to
        pre-process the labels `y_true` with the
        :func:`deel.lip.utils.process_labels_for_multi_gpu()` function.

        Args:
            alpha: regularization factor
            min_margin: positive float, minimum bound and initialization for margins.
            max_margin: positive float, minimum bound for margins.
            alpha_margin: regularization factor for margins
            (0.1 inforce that 90% of samples to be outside the margin).
                Kantorovich-Rubinstein term of the loss. In order to be consistent
                between hinge and KR, the first label must yield the positive class
                while the second yields negative class.
            multi_gpu (bool): set to True when running on multi-GPU/TPU
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        """
        self.max_margin = max_margin
        self.alpha_margin = alpha_margin
        self.hingeloss = HingeMarginAuto(
            min_margin=min_margin,
            max_margin=self.max_margin,
            alpha_margin=self.alpha_margin,
        )
        super(HKRauto, self).__init__(
            alpha=alpha,
            min_margin=min_margin,
            multi_gpu=multi_gpu,
            reduction=reduction,
            name=name,
        )

    def get_trainable_variables(self):
        return self.hingeloss.get_trainable_variables()

    @property
    def hinge_margins(self):
        return self.hingeloss.margins

    def get_config(self):
        config = {
            "max_margin": self.max_margin,
            "alpha_margin": self.alpha_margin,
        }
        base_config = super(HKRauto, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
        :func:`deel.lip.utils.process_labels_for_multi_gpu()` function.

        Args:
            min_margin: positive float, margin to enforce.
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        """
        self.min_margin = tf.Variable(min_margin, dtype=tf.float32)
        super(HingeMargin, self).__init__(reduction=reduction, name=name)

    @tf.function
    def compute_hinge_margin(self, y_true, y_pred, v_margin):
        sign = tf.where(y_true > 0, 1, -1)
        sign = tf.cast(sign, y_pred.dtype)
        hinge = tf.nn.relu(v_margin / 2.0 - sign * y_pred)
        # In binary case (`y_true` of shape (batch_size, 1)), `tf.reduce_mean(axis=-1)`
        # behaves like `tf.squeeze` to return element-wise loss of shape (batch_size, ).
        return tf.reduce_mean(hinge, axis=-1)

    def call(self, y_true, y_pred):
        return self.compute_hinge_margin(y_true, y_pred, self.min_margin)

    def get_config(self):
        config = {
            "min_margin": self.min_margin.numpy(),
        }
        base_config = super(HingeMargin, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "HingeMarginAuto")
class HingeMarginAuto(HingeMargin):
    def __init__(
        self,
        min_margin=1.0,
        max_margin=200,
        alpha_margin=0.1,
        reduction=Reduction.AUTO,
        name="HingeMarginAuto",
    ):
        r"""
        Compute the hinge margin loss.

        .. math::
            \underset{\textbf{x}}{\mathbb{E}} \left(\text{min_margin}
            -Yf(\textbf{x})\right)_+

        Note that `y_true` and `y_pred` must be of rank 2: (batch_size, 1) or
        (batch_size, C) for multilabel classification (with C categories).
        `y_true` accepts label values in (0, 1), (-1, 1), or pre-processed with the
        :func:`deel.lip.utils.process_labels_for_multi_gpu()` function.

        Args:
            min_margin: positive float, minimum bound and initialization for margins.
            max_margin: positive float, minimum bound for margins.
            alpha_margin: regularization factor for margins
            (0.1 inforce that 90% of samples to be outside the margin).
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        """
        self.alpha_margin = tf.Variable(alpha_margin, dtype=tf.float32)
        self.max_margin = max_margin
        self.min_margin_v = min_margin

        self.margins = None
        self.trainable_vars = None

        super(HingeMarginAuto, self).__init__(
            min_margin=min_margin, reduction=reduction, name=name
        )

    @tf.function
    def call(self, y_true, y_pred):
        if self.margins is None:
            self.margins = tf.Variable(
                np.array([self.min_margin_v] * y_true.shape[-1]),
                dtype=tf.float32,
                constraint=lambda x: tf.clip_by_value(
                    x, self.min_margin_v, self.max_margin
                ),
            )
            self.trainable_vars = [self.margins]

        hinge_value = self.compute_hinge_margin(y_true, y_pred, self.margins)

        regul_margin = tf.reduce_sum(self.margins)

        return hinge_value - self.alpha_margin * regul_margin

    def get_trainable_variables(self):
        return self.trainable_vars

    def get_config(self):
        config = {
            "alpha_margin": self.alpha_margin.numpy(),
            "max_margin": self.max_margin,
        }
        base_config = super(HingeMarginAuto, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "MulticlassKR")
class MulticlassKR(Loss):
    def __init__(self, multi_gpu=False, reduction=Reduction.AUTO, name="MulticlassKR"):
        r"""
        Loss to estimate average of Wasserstein-1 distance using Kantorovich-Rubinstein
        duality over outputs. In this multiclass setup, the KR term is computed for each
        class and then averaged.

        Note that `y_true` should be one-hot encoded or pre-processed with the
        :func:`deel.lip.utils.process_labels_for_multi_gpu()` function.

        Using a multi-GPU/TPU strategy requires to set `multi_gpu` to True and to
        pre-process the labels `y_true` with the
        :func:`deel.lip.utils.process_labels_for_multi_gpu()` function.

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


@register_keras_serializable("deel-lip", "MulticlassHinge")
class MulticlassHinge(Loss):
    def __init__(
        self,
        min_margin=1.0,
        soft_hinge_tau=0.0,
        reduction=Reduction.AUTO,
        name="MulticlassHinge",
    ):
        """
        Loss to estimate the Hinge loss in a multiclass setup. It computes the
        element-wise Hinge term. Note that this formulation differs from the one
        commonly found in tensorflow/pytorch (which maximises the difference between
        the two largest logits). This formulation is consistent with the binary
        classification loss used in a multiclass fashion.

        Note that `y_true` should be one-hot encoded or pre-processed with the
        :func:`deel.lip.utils.process_labels_for_multi_gpu()` function.

        Args:
            min_margin: positive float, margin to enforce.
            soft_hinge_tau: temperature applied in softmax for soft_hinge
              (default: set to 0 for classical hinge)
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        """
        self.min_margin = tf.Variable(min_margin, dtype=tf.float32)
        self.use_soft_hinge = soft_hinge_tau > 0.0
        self.soft_hinge_tau = tf.Variable(soft_hinge_tau, dtype=tf.float32)

        super(MulticlassHinge, self).__init__(reduction=reduction, name=name)

    @tf.function
    def compute_hinge_margin(self, y_true, y_pred, v_margin):

        # binary GT to support weighted y_true values
        # generated by process_labels_for_multi_gpu
        y_gt = tf.where(y_true > 0, 1, 0)
        y_gt = tf.cast(y_gt, y_pred.dtype)

        # Retrieve the y_pred value of the y_gt neuron for each sample
        vYtrue = tf.reduce_sum(y_pred * y_gt, axis=1, keepdims=True)
        # Compute centered hinge (m/2-vYtrue)+(m/2+y_pred)
        # for each output neuron and each sample
        hinge = tf.nn.relu(v_margin / 2.0 - vYtrue) + tf.nn.relu(
            v_margin / 2.0 + y_pred
        )
        if not self.use_soft_hinge:
            # Classical hinge: each neuron has the same weight
            # Discard y_gt neurons hinge values
            hinge = (1 - y_gt) * hinge
            # Divide by number of neurons to emulate reduce_mean
            # with the following reduce_sum
            if (len(tf.shape(y_pred)) == 2) and (tf.shape(y_pred)[-1] != 1):
                factor = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
            else:
                factor = 1.0
            # to compensate previous reduce_mean
            hinge = hinge / factor
        else:
            # Soft_hinge
            F_soft = tf.nn.softmax(
                tf.where(y_gt > 0, -tf.float32.max, (y_pred) * self.soft_hinge_tau),
                axis=-1,
            )
            hinge = hinge * F_soft

        return tf.reduce_sum(hinge, axis=-1)

    def call(self, y_true, y_pred):
        return self.compute_hinge_margin(y_true, y_pred, self.min_margin)

    def get_config(self):
        config = {
            "min_margin": self.min_margin.numpy(),
            "soft_hinge_tau": self.soft_hinge_tau.numpy(),
        }
        base_config = super(MulticlassHinge, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "MulticlassHingeAuto")
class MulticlassHingeAuto(MulticlassHinge):
    def __init__(
        self,
        min_margin=1.0,
        max_margin=200,
        alpha_margin=0.1,
        soft_hinge_tau=0.0,
        reduction=Reduction.AUTO,
        name="MulticlassHingeAuto",
    ):
        """
        Loss to estimate the Hinge loss in a multiclass setup. It computes the
        element-wise Hinge term. Note that this formulation differs from the one
        commonly found in tensorflow/pytorch (which maximises the difference between
        the two largest logits). This formulation is consistent with the binary
        classification loss used in a multiclass fashion.

        Note that `y_true` should be one-hot encoded or pre-processed with the
        :func:`deel.lip.utils.process_labels_for_multi_gpu()` function.

        Args:
            min_margin: positive float, minimum bound and initialization for margins.
            max_margin: positive float, minimum bound for margins.
            alpha_margin: regularization factor for margins
            (0.1 inforce that 90% of samples to be outside the margin).
            soft_hinge_tau: temperature applied in softmax for soft_hinge
              (default: set to 0 for classical hinge)
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        """
        self.alpha_margin = tf.Variable(alpha_margin, dtype=tf.float32)
        self.max_margin = max_margin
        self.min_margin_v = min_margin

        self.margins = None
        self.trainable_vars = None

        super(MulticlassHingeAuto, self).__init__(
            min_margin=min_margin,
            soft_hinge_tau=soft_hinge_tau,
            reduction=reduction,
            name=name,
        )

    @tf.function
    def call(self, y_true, y_pred):
        if self.margins is None:
            self.margins = tf.Variable(
                np.array([self.min_margin_v] * y_true.shape[-1]),
                dtype=tf.float32,
                constraint=lambda x: tf.clip_by_value(
                    x, self.min_margin_v, self.max_margin
                ),
            )
            self.trainable_vars = [self.margins]

        # binary GT to support weighted y_true values
        # generated by process_labels_for_multi_gpu
        y_gt = tf.where(y_true > 0, 1, 0)
        y_gt = tf.cast(y_gt, y_pred.dtype)

        # set the margin of the y_true class for each sample
        v_margin = tf.reduce_sum(self.margins * y_gt, axis=1, keepdims=True)

        hinge_value = self.compute_hinge_margin(y_true, y_pred, v_margin)

        # keep only neurons that have at least one sample of the class
        real_classes = tf.reduce_max(y_gt, axis=0)
        # compute the average margin over the classes
        # with at least one sample in the batch
        regul_margin = (
            1 / tf.reduce_sum(real_classes) * tf.reduce_sum(self.margins * real_classes)
        )

        return hinge_value - self.alpha_margin * regul_margin

    def get_trainable_variables(self):
        return self.trainable_vars

    def get_config(self):
        config = {
            "alpha_margin": self.alpha_margin.numpy(),
            "max_margin": self.max_margin,
        }
        base_config = super(MulticlassHingeAuto, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "MulticlassHKR")
class MulticlassHKR(Loss):
    def __init__(
        self,
        alpha=10.0,
        min_margin=1.0,
        soft_hinge_tau=0.0,
        multi_gpu=False,
        reduction=Reduction.AUTO,
        name="MulticlassHKR",
    ):
        """
        The multiclass version of HKR. This is done by computing the HKR term over each
        class and averaging the results.

        Note that `y_true` should be one-hot encoded or pre-processed with the
        :func:`deel.lip.utils.process_labels_for_multi_gpu()` function.

        Using a multi-GPU/TPU strategy requires to set `multi_gpu` to True and to
        pre-process the labels `y_true` with the
        :func:`deel.lip.utils.process_labels_for_multi_gpu()` function.

        Args:
            alpha: regularization factor
            min_margin: positive float, margin to enforce.
            soft_hinge_tau: temperature applied in softmax for soft_hinge
              (default: set to 0 for classical hinge)
            multi_gpu (bool): set to True when running on multi-GPU/TPU
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        """
        self.alpha = tf.Variable(alpha, dtype=tf.float32)
        self.min_margin = min_margin
        self.soft_hinge_tau = soft_hinge_tau
        self.multi_gpu = multi_gpu
        if not hasattr(self, "hingeloss"):  # compatibility with MulticlassHKRauto
            self.hingeloss = MulticlassHinge(
                self.min_margin, soft_hinge_tau=self.soft_hinge_tau
            )

        self.KRloss = MulticlassKR(multi_gpu=multi_gpu, reduction=reduction, name=name)
        if alpha == np.inf:  # alpha = inf => hinge only
            self.fct = self.hingeloss
        else:
            self.fct = self.hkr
        super(MulticlassHKR, self).__init__(reduction=reduction, name=name)

    @tf.function
    def hkr(self, y_true, y_pred):
        a = -self.KRloss.call(y_true, y_pred)
        b = self.hingeloss.call(y_true, y_pred)
        return a + self.alpha * b

    def call(self, y_true, y_pred):
        return self.fct(y_true, y_pred)

    def get_config(self):
        config = {
            "alpha": self.alpha.numpy(),
            "min_margin": self.min_margin,
            "soft_hinge_tau": self.soft_hinge_tau,
            "multi_gpu": self.multi_gpu,
        }
        base_config = super(MulticlassHKR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "MulticlassHKRauto")
class MulticlassHKRauto(MulticlassHKR):
    def __init__(
        self,
        alpha=10.0,
        min_margin=1.0,
        max_margin=200,
        alpha_margin=0.1,
        soft_hinge_tau=0.0,
        multi_gpu=False,
        reduction=Reduction.AUTO,
        name="MulticlassHKRauto",
    ):
        """
        The multiclass version of HKR. This is done by computing the HKR term over each
        class and averaging the results.

        Note that `y_true` should be one-hot encoded or pre-processed with the
        :func:`deel.lip.utils.process_labels_for_multi_gpu()` function.

        Using a multi-GPU/TPU strategy requires to set `multi_gpu` to True and to
        pre-process the labels `y_true` with the
        :func:`deel.lip.utils.process_labels_for_multi_gpu()` function.

        Args:
            alpha: regularization factor
            min_margin: positive float, minimum bound and initialization for margins.
            max_margin: positive float, minimum bound for margins.
            alpha_margin: regularization factor for margins
                (0.1 inforce that 90% of samples to be outside the margin).
            soft_hinge_tau: temperature applied in softmax for soft_hinge
              (default: set to 0 for classical hinge)
            multi_gpu (bool): set to True when running on multi-GPU/TPU
            reduction: passed to tf.keras.Loss constructor
            name: passed to tf.keras.Loss constructor

        """
        self.max_margin = max_margin
        self.alpha_margin = alpha_margin
        self.hingeloss = MulticlassHingeAuto(
            min_margin=min_margin,
            max_margin=self.max_margin,
            alpha_margin=self.alpha_margin,
            soft_hinge_tau=soft_hinge_tau,
        )
        super(MulticlassHKRauto, self).__init__(
            alpha=alpha,
            min_margin=min_margin,
            soft_hinge_tau=soft_hinge_tau,
            multi_gpu=multi_gpu,
            reduction=reduction,
            name=name,
        )

    def get_trainable_variables(self):
        return self.hingeloss.get_trainable_variables()

    @property
    def hinge_margins(self):
        return self.hingeloss.margins

    def get_config(self):
        config = {
            "max_margin": self.max_margin,
            "alpha_margin": self.alpha_margin,
        }
        base_config = super(MulticlassHKRauto, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "MultiMargin")
class MultiMargin(Loss):
    def __init__(self, min_margin=1.0, reduction=Reduction.AUTO, name="MultiMargin"):
        """
        Compute the hinge margin loss for multiclass (equivalent to Pytorch
        multi_margin_loss)

        Note that `y_true` should be one-hot encoded or pre-processed with the
        :func:`deel.lip.utils.process_labels_for_multi_gpu()` function.

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
    def __init__(
        self, min_margin, reduction=Reduction.AUTO, name="TauCategoricalCrossentropy"
    ):
        """
        Similar to original categorical hinge, but with a settable margin
        parameter.

        Args:
            min_margin: margin parameter.
            reduction: reduction of the loss, passed to original loss.
            name: name of the loss
        """
        self.min_margin = tf.Variable(min_margin, dtype=tf.float32)
        super(CategoricalHinge, self).__init__(name=name, reduction=reduction)

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        pos = tf.reduce_sum(y_true * y_pred, axis=-1)
        neg = tf.reduce_max((1.0 - y_true) * y_pred, axis=-1)
        zero = tf.cast(0.0, y_pred.dtype)
        return tf.maximum(neg - pos + self.min_margin, zero)

    def get_config(self):
        config = {
            "min_margin": self.min_margin.numpy(),
        }
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
        config = {
            "tau": self.tau.numpy(),
        }
        base_config = super(TauCategoricalCrossentropy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
