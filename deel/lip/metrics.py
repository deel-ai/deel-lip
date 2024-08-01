# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains metrics applicable in provable robustness. See
[https://arxiv.org/abs/2006.06520](https://arxiv.org/abs/2006.06520)
and [https://arxiv.org/abs/2108.04062](https://arxiv.org/abs/2108.04062) for more
information.
"""
import math
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.utils import register_keras_serializable


def _delta_multiclass(y_true, y_pred):
    r"""
    Compute the non-normalized provable robustness factor in multiclass setup (need
    to be adjusted with
    lipschitz constant.

    $$
    \Delta(x) = f_l(x) - \max_{i \neq l} f_i(x)
    $$

    Args:
        y_true: true labels must be in {1,0} or in {1,-1} (no label smoothing allowed)
        y_pred: network predictions

    Returns: non-normalized provable robustness factor

    """
    ynl_shape = (-1, tf.shape(y_pred)[-1] - 1)
    yl = tf.boolean_mask(y_pred, y_true > 0)
    ynl = tf.reshape(
        tf.boolean_mask(y_pred, y_true <= 0),
        ynl_shape,
    )
    delta = yl - tf.reduce_max(ynl, axis=-1)
    return delta


def _delta_binary(y_true, y_pred):
    r"""
    Compute the non-normalized provable robustness factor in binary setup (need to be
    adjusted with
    lipschitz constant).

    $$
    \Delta(x) = f(x) \text{ if } l=1, -f(x) \text{ otherwise}
    $$

    Args:
        y_true: true labels must be in {1,0} or in {1,-1} (no label smoothing allowed)
        y_pred: network predictions

    Returns: non-normalized provable robustness factor

    """
    y_true = tf.sign(tf.cast(y_true, y_pred.dtype) - 1e-3)
    return tf.multiply(y_true, y_pred)


@register_keras_serializable("deel-lip", "CategoricalProvableRobustAccuracy")
class CategoricalProvableRobustAccuracy(Loss):
    def __init__(
        self,
        epsilon=36 / 255,
        lip_const=1.0,
        disjoint_neurons=True,
        reduction="sum_over_batch_size",
        name="CategoricalProvableRobustAccuracy",
    ):
        r"""

        The accuracy that can be proved at a given epsilon.

        Args:
            epsilon (float): the metric will return the guaranteed accuracy for the
                radius epsilon.
            lip_const (float): lipschitz constant of the network
            disjoint_neurons (bool): must be set to True if your model ends with a
                FrobeniusDense layer with `disjoint_neurons` set to True. Set to False
                otherwise
            reduction: the recution method when training in a multi-gpu / TPU system
            name (str): metrics name.
        """
        self.lip_const = lip_const
        self.epsilon = epsilon
        self.disjoint_neurons = disjoint_neurons
        if disjoint_neurons:
            self.certificate_factor = 2 * lip_const
        else:
            self.certificate_factor = math.sqrt(2) * lip_const
        super(CategoricalProvableRobustAccuracy, self).__init__(
            reduction=reduction, name=name
        )

    @tf.function
    def call(self, y_true, y_pred):
        return tf.cast(
            (_delta_multiclass(y_true, y_pred) / self.certificate_factor)
            > self.epsilon,
            y_pred.dtype,
        )

    def get_config(self):
        config = {
            "epsilon": self.epsilon,
            "lip_const": self.lip_const,
            "disjoint_neurons": self.disjoint_neurons,
        }
        base_config = super(CategoricalProvableRobustAccuracy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "BinaryProvableRobustAccuracy")
class BinaryProvableRobustAccuracy(Loss):
    def __init__(
        self,
        epsilon=36 / 255,
        lip_const=1.0,
        reduction="sum_over_batch_size",
        name="BinaryProvableRobustAccuracy",
    ):
        r"""

        The accuracy that can be proved at a given epsilon.

        Args:
            epsilon (float): the metric will return the guaranteed accuracy for the
                radius epsilon.
            lip_const (float): lipschitz constant of the network
            reduction: the recution method when training in a multi-gpu / TPU system
            name (str): metrics name.
        """
        self.lip_const = lip_const
        self.epsilon = epsilon
        super(BinaryProvableRobustAccuracy, self).__init__(
            reduction=reduction, name=name
        )

    @tf.function
    def call(self, y_true, y_pred):
        return tf.cast(
            (_delta_binary(y_true, y_pred) / self.lip_const) > self.epsilon,
            y_pred.dtype,
        )

    def get_config(self):
        config = {
            "epsilon": self.epsilon,
            "lip_const": self.lip_const,
        }
        base_config = super(BinaryProvableRobustAccuracy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "CategoricalProvableAvgRobustness")
class CategoricalProvableAvgRobustness(Loss):
    def __init__(
        self,
        lip_const=1.0,
        disjoint_neurons=True,
        negative_robustness=False,
        reduction="sum_over_batch_size",
        name="CategoricalProvableAvgRobustness",
    ):
        r"""

        Compute the average provable robustness radius on the dataset.

        $$
        \mathbb{E}_{x \in D}\left[ \frac{\phi\left(\mathcal{M}_f(x)\right)}{L_f}\right]
        $$

        $\mathcal{M}_f(x)$ is a term that: is positive when x is correctly
        classified and negative otherwise. In both case the value give the robustness
        radius around x.

        In the multiclass setup we have:

        $$
        \mathcal{M}_f(x) =f_l(x) - \max_{i \neq l} f_i(x)
        $$

        Where $D$ is the dataset, $l$ is the correct label for x and
        $L_f$ is the lipschitz constant of the network ($L = 2 \times
        \text{lip_const}$ when `disjoint_neurons=True`, $L = \sqrt{2} \times
        \text{lip_const}$ otherwise).

        When `negative_robustness` is set to `True` misclassified elements count as
        negative robustness ($\phi$ act as identity function), when set to
        `False`,
        misclassified elements yield a robustness radius of 0 ( $\phi(x)=relu(
        x)$ ). The elements are not ignored when computing the mean in both cases.

        This metric works for labels both in {1,0} and {1,-1}.

        Args:
            lip_const (float): lipschitz constant of the network
            disjoint_neurons (bool): must be set to True is your model ends with a
                FrobeniusDense layer with `disjoint_neurons` set to True. Set to False
                otherwise
            reduction: the recution method when training in a multi-gpu / TPU system
            name (str): metrics name.
        """
        self.lip_const = lip_const
        self.disjoint_neurons = disjoint_neurons
        self.negative_robustness = negative_robustness
        if disjoint_neurons:
            self.certificate_factor = 2 * lip_const
        else:
            self.certificate_factor = math.sqrt(2) * lip_const
        if self.negative_robustness:
            self.delta_correction = lambda delta: delta
        else:
            self.delta_correction = tf.nn.relu
        super(CategoricalProvableAvgRobustness, self).__init__(
            reduction=reduction, name=name
        )

    @tf.function
    def call(self, y_true, y_pred):
        return (
            self.delta_correction(_delta_multiclass(y_true, y_pred))
            / self.certificate_factor
        )

    def get_config(self):
        config = {
            "lip_const": self.lip_const,
            "disjoint_neurons": self.disjoint_neurons,
            "negative_robustness": self.negative_robustness,
        }
        base_config = super(CategoricalProvableAvgRobustness, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "BinaryProvableAvgRobustness")
class BinaryProvableAvgRobustness(Loss):
    def __init__(
        self,
        lip_const=1.0,
        negative_robustness=False,
        reduction="sum_over_batch_size",
        name="BinaryProvableAvgRobustness",
    ):
        r"""

        Compute the average provable robustness radius on the dataset.

        $$
        \mathbb{E}_{x \in D}\left[ \frac{\phi\left(\mathcal{M}_f(x)\right)}{L_f}\right]
        $$

        $\mathcal{M}_f(x)$ is a term that: is positive when x is correctly
        classified and negative otherwise. In both case the value give the robustness
        radius around x.

        In the binary classification setup we have:

        $$
        \mathcal{M}_f(x) = f(x) \text{ if } l=1, -f(x) \text{otherwise}
        $$

        Where $D$ is the dataset, $l$ is the correct label for x and
        $L_f$ is the lipschitz constant of the network..

        When `negative_robustness` is set to `True` misclassified elements count as
        negative robustness ($\phi$ act as identity function), when set to
        `False`,
        misclassified elements yield a robustness radius of 0 ( $\phi(x)=relu(
        x)$ ). The elements are not ignored when computing the mean in both cases.

        This metric works for labels both in {1,0} and {1,-1}.

        Args:
            lip_const (float): lipschitz constant of the network
            reduction: the recution method when training in a multi-gpu / TPU system
            name (str): metrics name.
        """
        self.lip_const = lip_const
        self.negative_robustness = negative_robustness
        if self.negative_robustness:
            self.delta_correction = lambda delta: delta
        else:
            self.delta_correction = tf.nn.relu
        super(BinaryProvableAvgRobustness, self).__init__(
            reduction=reduction, name=name
        )

    @tf.function
    def call(self, y_true, y_pred):
        return self.delta_correction(_delta_binary(y_true, y_pred)) / self.lip_const

    def get_config(self):
        config = {
            "lip_const": self.lip_const,
            "negative_robustness": self.negative_robustness,
        }
        base_config = super(BinaryProvableAvgRobustness, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
