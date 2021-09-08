# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains metrics applicable in provable robustness. See
https://arxiv.org/abs/2006.06520 for more information.
"""
import math
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import Reduction
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable("deel-lip", "ProvableRobustness")
class ProvableRobustness(tf.keras.losses.Loss):
    def __init__(
        self,
        lip_const=1.0,
        disjoint_neurons=True,
        reduction=Reduction.AUTO,
        name="ProvableRobustness",
    ):
        r"""
        Compute the provable robustness, defined by :

        .. math::
            cert(x) = \frac{top_1 (y_i) - top_2(y_i)}{2l}


        when `disjoint_neurons` is set to True, or

        ..math::
            cert(x) = \frac{top_1 (y_i) - top_2(y_i)}{l\sqrt{2}}


        When `disjoint_neurons` is set to false.


        Where l is the lipschitz constant of the network. Note that when your model
        ends with a FrobeniusDense layer with parameter `disjoint_neurons=True` the
        computation is slightly different ( see refences for more information ).

        Notes:
            This loss differs doesn't need y_true label to be computed.

        References:
            Serrurier et al. https://arxiv.org/abs/2006.06520

        Args:
            lip_const: lipschitz constant of the network
            disjoint_neurons: must be set to True is your model ends with a
                FrobeniusDense layer with `disjoint_neurons` set to True. Set to False
                otherwise
            name: metrics name.
            **kwargs: parameters passed to the tf.keras.Loss constructor
        """
        self.lip_const = lip_const
        self.disjoint_neurons = disjoint_neurons
        if disjoint_neurons:
            self.certificate_factor = 2 * lip_const
        else:
            self.certificate_factor = math.sqrt(2) * lip_const
        super(ProvableRobustness, self).__init__(reduction, name)

    def call(self, y_true, y_pred):
        values, classes = tf.math.top_k(y_pred, k=2)
        avg_robustness = tf.reduce_mean(
            (values[:, 0] - values[:, 1]) / self.certificate_factor
        )
        return avg_robustness

    def get_config(self):
        config = {
            "lip_const": self.lip_const,
            "disjoint_neurons": self.disjoint_neurons,
        }
        base_config = super(ProvableRobustness, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "AdjustedRobustness")
class AdjustedRobustness(Loss):
    def __init__(
        self,
        lip_const=1.0,
        disjoint_neurons=True,
        reduction=Reduction.AUTO,
        name="AdjustedRobustness",
    ):
        r"""
        Compute the adjusted robustness, defined by :

        .. math::
            cert_{acc}(x) = \frac{y_{i=label} - \max_{i \neq label} y_i}{2l}

        when `disjoint_neurons` is set to True, or

        .. math::
            cert_{acc}(x) = \frac{y_{i=label} - \max_{i \neq label} y_i}{l\sqrt{2}}

        When `disjoint_neurons` is set to false.


        Where l is the lipschitz constant of the network. Note that when your model
        ends with a FrobeniusDense layer with parameter `disjoint_neurons=True` the
        computation is slightly different ( see refences for more information ).

        Notes:
            This loss differs from ProvableRobustness loss as a misclassification
            from the model yield a negative certificate (this require the knowledge
            of true labels)

        References:
            Serrurier et al. https://arxiv.org/abs/2006.06520

        Args:
            lip_const: lipschitz constant of the network.
            disjoint_neurons: must be set to True is your model ends with a
                FrobeniusDense layer with `disjoint_neurons` set to True. Set to False
                otherwise
            name: metrics name.
            **kwargs: parameters passed to the tf.keras.Loss constructor
        """
        self.lip_const = lip_const
        self.disjoint_neurons = disjoint_neurons
        if disjoint_neurons:
            self.certificate_factor = 2 * lip_const
        else:
            self.certificate_factor = math.sqrt(2) * lip_const
        super(AdjustedRobustness, self).__init__(reduction, name)

    def call(self, y_true, y_pred):
        values, classes = tf.math.top_k(y_pred, k=2)
        robustness_bound = tf.where(
            tf.argmax(y_pred, axis=-1) == tf.argmax(y_true, axis=-1),
            (values[:, 0] - values[:, 1]) / self.certificate_factor,
            -(values[:, 0] - values[:, 1]) / self.certificate_factor
            # mislabelling => negative robustness
        )
        avg_robustness = tf.reduce_mean(robustness_bound)
        return avg_robustness

    def get_config(self):
        config = {
            "lip_const": self.lip_const,
            "disjoint_neurons": self.disjoint_neurons,
        }
        base_config = super(AdjustedRobustness, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
