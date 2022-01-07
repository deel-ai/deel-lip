# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains metrics applicable in provable robustness. See
https://arxiv.org/abs/2006.06520 and https://arxiv.org/abs/2108.04062 for more
information.
"""
import math
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import Reduction
from tensorflow.keras.utils import register_keras_serializable


def _delta_multiclass(y_true, y_pred):
    r"""
    Compute the non-normalized provable robustness factor in multiclass setup (need
    to be adjusted with
    lipschitz constant.

    .. math::
            \Delta(x) = f_l(x) - \max_{i \neq l} f_i(x)

    Args:
        y_true: true labels must be in {1,0} or in {1,-1} (no label smoothing allowed)
        y_pred: network predictions

    Returns: non-normalized provable robustness factor

    """
    ynl_shape = (-1, tf.shape(y_pred)[-1] - 1)
    yl = tf.boolean_mask(y_pred, y_true == 1)
    ynl = tf.reshape(
        tf.boolean_mask(y_pred, y_true != 1),
        ynl_shape,
    )
    delta = yl - tf.reduce_max(ynl, axis=-1)
    return delta


def _delta_binary(y_true, y_pred):
    r"""
    Compute the non-normalized provable robustness factor in binary setup (need to be
    adjusted with
    lipschitz constant).

    .. math::
            \Delta(x) = f(x) \text{ if } l=1, -f(x) \text{ otherwise}

    Args:
        y_true: true labels must be in {1,0} or in {1,-1} (no label smoothing allowed)
        y_pred: network predictions

    Returns: non-normalized provable robustness factor

    """
    y_true = tf.sign(tf.cast(y_true, y_pred.dtype) - 1e-3)
    return tf.multiply(y_true, y_pred)


@register_keras_serializable("deel-lip", "ProvableRobustAccuracy")
class ProvableRobustAccuracy(Loss):
    def __init__(
        self,
        epsilon=36 / 255,
        lip_const=1.0,
        disjoint_neurons=True,
        reduction=Reduction.AUTO,
        name="ProvableRobustAccuracy",
    ):
        r"""

        The accuracy that can be proved at a given epsilon.

        Args:
            epsilon: the metric will return the guaranteed accuracy for the radius
                epsilon
            lip_const: lipschitz constant of the network
            disjoint_neurons: must be set to True if your model ends with a
                FrobeniusDense layer with `disjoint_neurons` set to True. Set to False
                otherwise
            reduction: the recution method when training in a multi-gpu / TPU system
            name: metrics name.
        """
        self.lip_const = lip_const
        self.epsilon = epsilon
        self.disjoint_neurons = disjoint_neurons
        if disjoint_neurons:
            self.certificate_factor = 2 * lip_const
        else:
            self.certificate_factor = math.sqrt(2) * lip_const
        super(ProvableRobustAccuracy, self).__init__(reduction, name)

    def call(self, y_true, y_pred):
        shape = y_true.shape
        if len(shape) == 2 and (shape[-1] > 1):
            delta_fct = _delta_multiclass
        else:
            delta_fct = _delta_binary
            self.certificate_factor = self.lip_const
        return tf.reduce_mean(
            tf.cast(
                (delta_fct(y_true, y_pred) / self.certificate_factor) > self.epsilon,
                tf.float32,
            )
        )

    def get_config(self):
        config = {
            "epsilon": self.epsilon,
            "lip_const": self.lip_const,
            "disjoint_neurons": self.disjoint_neurons,
        }
        base_config = super(ProvableRobustAccuracy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "ProvableAvgRobustness")
class ProvableAvgRobustness(Loss):
    def __init__(
        self,
        lip_const=1.0,
        disjoint_neurons=True,
        negative_robustness=False,
        reduction=Reduction.AUTO,
        name="ProvableAvgRobustness",
    ):
        r"""

        Compute the average provable robustness radius on the dataset.

        .. math::
            \mathbb{E}_{x \in D}\left[ \frac{\phi\left(\mathcal{M}_f(x)\right)}{
            L_f}\right]

        :math:`\mathcal{M}_f(x)` is a term that: is positive when x is correctly
        classified and negative otherwise. In both case the value give the robustness
        radius around x.

        In the multiclass setup we have:

        .. math::
            \mathcal{M}_f(x) =f_l(x) - \max_{i \neq l} f_i(x)

        In the binary classification setup we have:

        .. math::
            \mathcal{M}_f(x) = f(x) \text{ if } l=1, -f(x) \text{otherwise}

        Where :math:`D` is the dataset, :math:`l` is the correct label for x and
        :math:`L_f` is the lipschitz constant of the network (:math:`L = 2 \times
        \text{lip_const}` when `disjoint_neurons=True`, :math:`L = \sqrt{2} \times
        \text{lip_const}` otherwise).

        When `negative_robustness` is set to `True` misclassified elements count as
        negative robustness (:math:`\phi` act as identity function), when set to
        `False`,
        misclassified elements yield a robustness radius of 0 ( :math:`\phi(x)=relu(
        x)` ). The elements are not ignored when computing the mean in both cases.

        This metric works for labels both in {1,0} and {1,-1}.

        Args:
            lip_const: lipschitz constant of the network
            disjoint_neurons: must be set to True is your model ends with a
                FrobeniusDense layer with `disjoint_neurons` set to True. Set to False
                otherwise
            reduction: the recution method when training in a multi-gpu / TPU system
            name: metrics name.
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
        super(ProvableAvgRobustness, self).__init__(reduction, name)

    def call(self, y_true, y_pred):
        shape = y_true.shape
        if len(shape) == 2 and (shape[-1] > 1):
            delta_fct = _delta_multiclass
        else:
            delta_fct = _delta_binary
            self.certificate_factor = self.lip_const
        return tf.reduce_mean(
            self.delta_correction(delta_fct(y_true, y_pred)) / self.certificate_factor
        )

    def get_config(self):
        config = {
            "lip_const": self.lip_const,
            "disjoint_neurons": self.disjoint_neurons,
            "negative_robustness": self.negative_robustness,
        }
        base_config = super(ProvableAvgRobustness, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
