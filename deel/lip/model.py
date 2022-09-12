# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains equivalents for Model and Sequential. These classes add support
for condensation and vanilla exportation.
"""
import warnings
import math
from warnings import warn
import numpy as np
from tensorflow.keras import Sequential as KerasSequential, Model as KerasModel
from .layers import LipschitzLayer, Condensable
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.models import clone_model
import tensorflow as tf


@register_keras_serializable("deel-lip", "Sequential")
class Sequential(KerasSequential, LipschitzLayer, Condensable):
    def __init__(
        self,
        layers=None,
        name=None,
        k_coef_lip=1.0,
    ):
        """
        Equivalent of keras.Sequential but allow to set k-lip factor globally. Also
        support condensation and vanilla exportation.
        For now constant repartition is implemented (each layer
        get n_sqrt(k_lip_factor), where n is the number of layers)
        But in the future other repartition function may be implemented.

        Args:
            layers: list of layers to add to the model.
            name: name of the model, can be None
            k_coef_lip: the Lipschitz coefficient to ensure globally on the model.
        """
        super(Sequential, self).__init__(layers, name)
        self.set_klip_factor(k_coef_lip)

    def build(self, input_shape=None):
        self._init_lip_coef(input_shape)
        return super(Sequential, self).build(input_shape)

    def set_klip_factor(self, klip_factor):
        super(Sequential, self).set_klip_factor(klip_factor)
        nb_layers = np.sum([isinstance(layer, LipschitzLayer) for layer in self.layers])
        for layer in self.layers:
            if isinstance(layer, LipschitzLayer):
                layer.set_klip_factor(math.pow(klip_factor, 1 / nb_layers))
            else:
                warn(
                    "Sequential model contains a layer wich is not a Lipschitz layer: {}".format(  # noqa: E501
                        layer.name
                    )
                )

    def _compute_lip_coef(self, input_shape=None):
        for layer in self.layers:
            if isinstance(layer, LipschitzLayer):
                layer._compute_lip_coef(input_shape)
            else:
                warn(
                    "Sequential model contains a layer wich is not a Lipschitz layer: {}".format(  # noqa: E501
                        layer.name
                    )
                )

    def _init_lip_coef(self, input_shape):
        for layer in self.layers:
            if isinstance(layer, LipschitzLayer):
                layer._init_lip_coef(input_shape)
            else:
                warn(
                    "Sequential model contains a layer wich is not a Lipschitz layer: {}".format(  # noqa: E501
                        layer.name
                    )
                )

    def _get_coef(self):
        global_coef = 1.0
        for layer in self.layers:
            if isinstance(layer, LipschitzLayer) and (global_coef is not None):
                global_coef *= layer._get_coef()
            else:
                warn(
                    "Sequential model contains a layer wich is not a Lipschitz layer: {}".format(  # noqa: E501
                        layer.name
                    )
                )
                global_coef = None
        return global_coef

    def condense(self):
        for layer in self.layers:
            if isinstance(layer, Condensable):
                layer.condense()

    def vanilla_export(self):
        return vanillaModel(self)

    def get_config(self):
        config = {"k_coef_lip": self.k_coef_lip}
        base_config = super(Sequential, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable("deel-lip", "Model")
class Model(KerasModel):
    """
    Equivalent of keras.Model but support condensation and vanilla exportation.

    Warning:
         As lipschitz constant are multiplicative along layer, the Model class
         cannot set a global Lipschitz constant (problem with branching inside a
         model).
    """

    def condense(self):
        for layer in self.layers:
            if isinstance(layer, Condensable):
                layer.condense()

    def vanilla_export(self) -> KerasModel:
        """
        Export this model to a "Vanilla" model, i.e. a model without Condensable
        layers.

        Returns:
            A Keras model, identical to this model, but where condensable layers have
            been replaced with their vanilla equivalent (e.g. SpectralConv2D with
             Conv2D).
        """
        return vanillaModel(self)


def vanillaModel(model):
    """
    Transform a model to its equivalent "vanilla" model, i.e. a model where
    `Condensable` layers are replaced with their vanilla equivalent. For example,
    `SpectralConv2D` layers are converted to tf.keras `Conv2D` layers.

    The input model can be a tf.keras Sequential/Model or a deel.lip Sequential/Model.

    Args:
        model: a tf.keras or deel.lip model with Condensable layers.

    Returns:
        A Keras model, identical to the input model where `Condensable` layers are
            replaced with their vanilla counterparts.
    """

    def _replace_condensable_layer(layer):
        # Return a vanilla layer if Condensable, else return a copy of the layer
        if isinstance(layer, Condensable):
            return layer.vanilla_export()
        new_layer = layer.__class__.from_config(layer.get_config())
        new_layer.build(layer.input_shape)
        new_layer.set_weights(layer.get_weights())
        return new_layer

    return clone_model(model, clone_function=_replace_condensable_layer)


def lossvariables_train_step(model, data):
    # Unpack the data. Its structure depends on your model and
    # on what you pass to `fit()`.
    if len(data) == 3:
        x, y, sample_weight = data
    else:
        sample_weight = None
        x, y = data

    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)  # Forward pass
        # Compute the loss value.
        # The loss function is configured in `compile()`.
        loss = model.compiled_loss(
            y,
            y_pred,
            sample_weight=sample_weight,
            regularization_losses=model.losses,
        )

    # Compute gradients
    trainable_vars = model.trainable_variables

    if hasattr(model.loss, "get_trainable_variables") and model.optim_margin:
        trainable_vars = trainable_vars + model.loss.get_trainable_variables()

    gradients = tape.gradient(loss, trainable_vars)
    # Update weights
    model.optimizer.apply_gradients(zip(gradients, trainable_vars))

    # Update the metrics.
    # Metrics are configured in `compile()`.
    model.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

    # Return a dict mapping metric names to current value.
    # Note that it will include the loss (tracked in self.metrics).
    return {m.name: m.result() for m in model.metrics}


class LossVariableModel(Model):
    def __init__(self, optim_margin=True, **kwargs):
        """
        Superclass for models. When training, will update
        trainable variables in the loss.

        Args:
            optim_margin : A flag to activate/deactivate
            loss trainable variables learning
        """
        super(LossVariableModel, self).__init__(**kwargs)
        self.optim_margin = optim_margin

    def compile(self, **kwargs):
        super(LossVariableModel, self).compile(**kwargs)
        if not hasattr(self.loss, "get_trainable_variables"):
            warnings.warn(
                "LossVariableModel: warning the loss has no trainable parameters."
            )

    @tf.function
    def train_step(self, data):
        return lossvariables_train_step(self, data)

    def get_config(self):
        config = {
            "optim_margin": self.optim_margin,
        }
        base_config = super(LossVariableModel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LossVariableSequential(Sequential):
    def __init__(self, layers=None, name=None, optim_margin=True, **kwargs):
        """
        Superclass for models. When training, will update
        trainable variables in the loss.

        Args:
            optim_margin : A flag to activate/deactivate
            loss trainable variables learning
        """
        super(LossVariableSequential, self).__init__(layers=layers, name=name, **kwargs)
        self.optim_margin = optim_margin

    def compile(self, **kwargs):
        super(LossVariableSequential, self).compile(**kwargs)
        if not hasattr(self.loss, "get_trainable_variables"):
            warnings.warn(
                "LossVariableModel: warning the loss has no trainable parameters."
            )

    @tf.function
    def train_step(self, data):
        return lossvariables_train_step(self, data)

    def get_config(self):
        config = {
            "optim_margin": self.optim_margin,
        }
        base_config = super(LossVariableSequential, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
