"""These tests assert that deel.lip Sequential and Model objects behave as expected."""

import warnings
from unittest import TestCase
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
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
        kl.Flatten(),
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
    x = kl.Add()([x, x1])
    x = kl.Flatten()(x)
    x = kl.Dense(4)(x)
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

    def test_warning_unsupported_1Lip_layers(self):
        """Assert that some unsupported layers return a warning message that they are
        not 1-Lipschitz and other supported layers don't raise a warning.
        """

        # Check that supported 1-Lipschitz layers do not raise a warning
        supported_layers = [
            kl.Input((32, 32, 3)),
            kl.ReLU(),
            kl.Activation("relu"),
            kl.Softmax(),
            kl.Flatten(),
            kl.Reshape((10,)),
            kl.MaxPool2D(),
            SpectralDense(3),
            ScaledL2NormPooling2D(),
        ]
        for lay in supported_layers:
            with warnings.catch_warnings(record=True) as w:
                Sequential([lay])
                self.assertEqual(len(w), 0, f"Layer {lay.name} shouldn't raise warning")

        # Check that unsupported layers raise a warning
        unsupported_layers = [
            kl.MaxPool2D(pool_size=3, strides=2),
            kl.Add(),
            kl.Concatenate(),
            kl.Activation("gelu"),
            kl.Dense(5),
            kl.Conv2D(10, 3),
            kl.UpSampling2D(),
        ]
        for lay in unsupported_layers:
            with self.assertWarnsRegex(
                Warning,
                expected_regex="layer which is not a 1-Lipschitz",
                msg=f"Layer {lay.name} should raise a warning.",
            ):
                Sequential([lay])
