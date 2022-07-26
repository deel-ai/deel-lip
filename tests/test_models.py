"""These tests assert that deel.lip Sequential and Model objects behave as expected."""

from unittest import TestCase
import numpy as np
import tensorflow as tf
from deel.lip import Sequential, Model, vanillaModel
from deel.lip.layers import SpectralConv2D, SpectralDense, ScaledL2NormPooling2D
from deel.lip.activations import GroupSort2
from deel.lip.model import LossVariableModel, LossVariableSequential


def sequential_layers():
    """Return list of layers for a Sequential model"""
    return [
        SpectralConv2D(6, 3, input_shape=(20, 20, 3)),
        ScaledL2NormPooling2D(),
        GroupSort2(),
        tf.keras.layers.Flatten(),
        SpectralDense(10),
    ]


def functional_input_output_tensors():
    """Return input and output tensor of a Functional (hard-coded) model"""
    inputs = tf.keras.Input((8, 8, 3))
    x = SpectralConv2D(2, (3, 3), k_coef_lip=2.0)(inputs)
    x = GroupSort2()(x)
    x = ScaledL2NormPooling2D((2, 2), k_coef_lip=2.0)(x)
    x1 = SpectralConv2D(2, (3, 3), k_coef_lip=2.0)(x)
    x1 = GroupSort2()(x1)
    x = tf.keras.layers.Add()([x, x1])
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4)(x)
    x = SpectralDense(4, k_coef_lip=2.0)(x)
    outputs = SpectralDense(2, k_coef_lip=2.0)(x)
    return inputs, outputs


class Test(TestCase):
    def assert_model_outputs(self, model1, model2):
        """Assert outputs are identical for both models on random inputs"""
        x = np.random.random((10,) + model1.input_shape[1:]).astype(np.float32)
        y1 = model1.predict(x)
        y2 = model2.predict(x)
        np.testing.assert_array_equal(y2, y1)

    def test_keras_Sequential(self):
        """Assert vanilla conversion of a tf.keras.Sequential model"""
        model = tf.keras.Sequential(sequential_layers())
        vanilla_model = vanillaModel(model)
        self.assert_model_outputs(model, vanilla_model)

    def test_deel_lip_Sequential(self):
        """Assert vanilla conversion of a deel.lip.Sequential model"""
        model = Sequential(sequential_layers())
        vanilla_model = model.vanilla_export()
        self.assert_model_outputs(model, vanilla_model)

    def test_keras_Model(self):
        """Assert vanilla conversion of a tf.keras.Model model"""
        inputs, outputs = functional_input_output_tensors()
        model = tf.keras.Model(inputs, outputs)
        vanilla_model = vanillaModel(model)
        self.assert_model_outputs(model, vanilla_model)

    def test_deel_lip_Model(self):
        """Assert vanilla conversion of a deel.lip.Model model"""
        inputs, outputs = functional_input_output_tensors()
        model = Model(inputs, outputs)
        vanilla_model = model.vanilla_export()
        self.assert_model_outputs(model, vanilla_model)

    def test_LossVariableModel(self):
        """Assert vanilla conversion of a deel.lip.LossVariableModel model"""
        inputs, outputs = functional_input_output_tensors()
        model = LossVariableModel(inputs=inputs, outputs=outputs)
        vanilla_model = model.vanilla_export()
        self.assert_model_outputs(model, vanilla_model)

    def test_LossVariableSequential(self):
        """Assert vanilla conversion of a deel.lip.LossVariableSequential model"""
        model = LossVariableSequential(sequential_layers())
        vanilla_model = model.vanilla_export()
        self.assert_model_outputs(model, vanilla_model)
