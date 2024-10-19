# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""These tests assert that deel.lip Sequential and Model objects behave as expected."""

import warnings
import pytest
import numpy as np
from collections import OrderedDict


from . import utils_framework as uft

from .utils_framework import (
    Sequential,
    Model,
    tSequential,
    tModel,
    vanillaModel,
    SpectralConv2d,
    SpectralLinear,
    ScaledL2NormPool2d,
    FrobeniusLinear,
    tInput,
)
from .utils_framework import GroupSort2, Flatten
from .utils_framework import (
    tAdd,
    tLinear,
    tReLU,
    tActivation,
    tSoftmax,
    tReshape,
    tMaxPool2d,
    tConcatenate,
    tConv2d,
    tUpSampling2d,
)


def sequential_layers(input_shape):
    """Return list of layers for a Sequential model"""
    return [
        uft.get_instance_framework(tInput, {"shape": input_shape}),  # (20, 20, 3)
        uft.get_instance_framework(
            SpectralConv2d,
            {"in_channels": 3, "out_channels": 6, "kernel_size": 3, "padding": 1},
        ),
        uft.get_instance_framework(ScaledL2NormPool2d, {"kernel_size": (2, 2)}),
        uft.get_instance_framework(GroupSort2, {}),
        uft.get_instance_framework(Flatten, {}),
        uft.get_instance_framework(
            SpectralLinear, {"in_features": 600, "out_features": 10}
        ),
    ]


def get_functional_tensors(input_shape):
    dict_functional_tensors = {}
    dict_functional_tensors["inputs"] = uft.get_instance_framework(
        tInput, {"shape": input_shape}
    )
    dict_functional_tensors["conv1"] = uft.get_instance_framework(
        SpectralConv2d,
        {
            "in_channels": 3,
            "out_channels": 2,
            "kernel_size": (3, 3),
            "k_coef_lip": 2.0,
            "padding": 1,
        },
    )
    dict_functional_tensors["act1"] = uft.get_instance_framework(GroupSort2, {})
    dict_functional_tensors["pool1"] = uft.get_instance_framework(
        ScaledL2NormPool2d, {"kernel_size": (2, 2), "k_coef_lip": 2.0}
    )
    dict_functional_tensors["conv2"] = uft.get_instance_framework(
        SpectralConv2d,
        {
            "in_channels": 2,
            "out_channels": 2,
            "kernel_size": (3, 3),
            "k_coef_lip": 2.0,
            "padding": 1,
        },
    )
    dict_functional_tensors["act2"] = uft.get_instance_framework(GroupSort2, {})
    dict_functional_tensors["add2"] = uft.get_instance_framework(tAdd, {})
    dict_functional_tensors["flatten"] = uft.get_instance_framework(Flatten, {})
    dict_functional_tensors["dense1"] = uft.get_instance_framework(
        tLinear, {"in_features": 32, "out_features": 4}
    )
    dict_functional_tensors["dense2"] = uft.get_instance_framework(
        SpectralLinear, {"in_features": 4, "out_features": 4, "k_coef_lip": 2.0}
    )
    dict_functional_tensors["dense3"] = uft.get_instance_framework(
        SpectralLinear, {"in_features": 4, "out_features": 2, "k_coef_lip": 2.0}
    )
    return dict_functional_tensors


def functional_input_output_tensors(dict_functional_tensors, x):
    """Return input and output tensor of a Functional (hard-coded) model"""
    if dict_functional_tensors["inputs"] is None:
        inputs = x
    else:
        inputs = dict_functional_tensors["inputs"]
    x = dict_functional_tensors["conv1"](inputs)
    x = dict_functional_tensors["act1"](x)
    x = dict_functional_tensors["pool1"](x)
    x1 = dict_functional_tensors["conv2"](x)
    x1 = dict_functional_tensors["act2"](x1)
    x = dict_functional_tensors["add2"]([x, x1])
    x = dict_functional_tensors["flatten"](x)
    x = dict_functional_tensors["dense1"](x)
    x = dict_functional_tensors["dense2"](x)
    outputs = dict_functional_tensors["dense3"](x)
    if dict_functional_tensors["inputs"] is None:
        return outputs
    else:
        return inputs, outputs
    # return x


# class Test(TestCase):
def assert_model_outputs(input_shape, model1, model2):
    """Assert outputs are identical for both models on random inputs"""
    x = np.random.random((10,) + input_shape).astype(np.float32)
    x = uft.to_tensor(x)
    y1 = uft.compute_predict(model1, x, training=False)  # .predict(x) #
    y2 = uft.compute_predict(model2, x, training=False)  # .predict(x) #model2(x)  #
    np.testing.assert_allclose(uft.to_numpy(y2), uft.to_numpy(y1), atol=1e-5)


def test_keras_Sequential():
    """Assert vanilla conversion of a tf.keras.Sequential model"""
    input_shape = uft.to_framework_channel((3, 20, 20))
    model = uft.generate_k_lip_model(
        tSequential,
        {"layers": sequential_layers(input_shape)},
        input_shape=input_shape,
        k=None,
    )
    if uft.vanilla_require_a_copy():
        model2 = uft.generate_k_lip_model(
            tSequential,
            {"layers": sequential_layers(input_shape)},
            input_shape=input_shape,
            k=None,
        )
        uft.copy_model_parameters(model, model2)
        vanilla_model = vanillaModel(model2)
    else:
        vanilla_model = vanillaModel(model)

    assert_model_outputs(input_shape, model, vanilla_model)


def test_deel_lip_Sequential():
    """Assert vanilla conversion of a deel.lip.Sequential model"""
    input_shape = uft.to_framework_channel((3, 20, 20))
    model = uft.generate_k_lip_model(
        Sequential, {"layers": sequential_layers(input_shape)}, input_shape=input_shape
    )
    if uft.vanilla_require_a_copy():
        model2 = uft.generate_k_lip_model(
            Sequential,
            {"layers": sequential_layers(input_shape)},
            input_shape=input_shape,
        )
        uft.copy_model_parameters(model, model2)
        vanilla_model = model2.vanilla_export()
    else:
        vanilla_model = model.vanilla_export()
    assert_model_outputs(input_shape, model, vanilla_model)


@pytest.mark.skipif(
    hasattr(tModel, "unavailable_class"),
    reason="tModel not available",
)
def test_Model():
    """Assert vanilla conversion of a tf.keras.Model model"""
    input_shape = uft.to_framework_channel((3, 8, 8))
    dict_tensors = get_functional_tensors(input_shape)
    model = uft.get_functional_model(
        tModel, dict_tensors, functional_input_output_tensors
    )
    # inputs, outputs = functional_input_output_tensors()
    #    model = tf.keras.Model(inputs, outputs)
    if uft.vanilla_require_a_copy():
        dict_tensors2 = get_functional_tensors(input_shape)
        model2 = uft.get_functional_model(
            tModel, dict_tensors2, functional_input_output_tensors
        )
        uft.copy_model_parameters(model, model2)
        vanilla_model = vanillaModel(model2)
    else:
        vanilla_model = vanillaModel(model)
    assert_model_outputs(input_shape, model, vanilla_model)


@pytest.mark.skipif(
    hasattr(Model, "unavailable_class"),
    reason="Model not available",
)
def test_lip_Model():
    """Assert vanilla conversion of a deel.lip.Model model"""
    input_shape = uft.to_framework_channel((3, 8, 8))
    dict_tensors = get_functional_tensors(input_shape)
    model = uft.get_functional_model(
        Model, dict_tensors, functional_input_output_tensors
    )
    # inputs, outputs = functional_input_output_tensors()
    # model = Model(inputs, outputs)
    if uft.vanilla_require_a_copy():
        dict_tensors2 = get_functional_tensors(input_shape)
        model2 = uft.get_functional_model(
            Model, dict_tensors2, functional_input_output_tensors
        )
        uft.copy_model_parameters(model, model2)
        vanilla_model = model2.vanilla_export()
    else:
        vanilla_model = model.vanilla_export()
    assert_model_outputs(input_shape, model, vanilla_model)


def test_warning_unsupported_1Lip_layers():
    """Assert that some unsupported layers return a warning message that they are
    not 1-Lipschitz and other supported layers don't raise a warning.
    """

    # Check that supported 1-Lipschitz layers do not raise a warning
    input_shape = uft.to_framework_channel((3, 32, 32))
    supported_layers = [
        uft.get_instance_framework(
            tInput, {"shape": input_shape}
        ),  # kl.tInput((32, 32, 3)),
        uft.get_instance_framework(tReLU, {}),  # kl.ReLU(),
        uft.get_instance_framework(
            tActivation, {"activation": "relu"}
        ),  # kl.Activation("relu"),
        uft.get_instance_framework(tSoftmax, {}),  # kl.Softmax(),
        uft.get_instance_framework(Flatten, {}),  # kl.Flatten(),
        uft.get_instance_framework(tReshape, {"target_shape": (10,)}),  # kl.Reshape(),
        uft.get_instance_framework(
            tMaxPool2d, {"kernel_size": (2, 2)}
        ),  # kl.MaxPool2d(),
        uft.get_instance_framework(
            SpectralLinear, {"in_features": 3, "out_features": 3}
        ),
        uft.get_instance_framework(
            ScaledL2NormPool2d, {"kernel_size": (2, 2)}
        ),  # ScaledL2NormPool2d(),
    ]
    for lay in supported_layers:
        with warnings.catch_warnings(record=True) as w:
            if lay is not None:
                _ = uft.generate_k_lip_model(
                    Sequential,
                    {"layers": [lay]},
                    input_shape=None,
                    k=None,
                )
                assert len(w) == 0, f"Layer {lay} shouldn't raise warning"

    # Check that unsupported layers raise a warning
    unsupported_layers = [
        uft.get_instance_framework(
            tMaxPool2d, {"kernel_size": 3, "stride": 2}
        ),  # kl.MaxPool2d(),
        uft.get_instance_framework(tAdd, {}),  # kl.Add(),
        uft.get_instance_framework(tConcatenate, {}),  # kl.Concatenate(),
        uft.get_instance_framework(
            tLinear, {"in_features": 5, "out_features": 5}
        ),  # kl.tLinear(5),
        uft.get_instance_framework(
            tConv2d, {"in_channels": 10, "out_channels": 10, "kernel_size": 3}
        ),  # kl.Conv2d(10, 3),
        uft.get_instance_framework(tUpSampling2d, {}),  # kl.UpSampling2d(),
        uft.get_instance_framework(
            tActivation, {"activation": "gelu"}
        ),  # kl.Activation("relu"),
    ]

    for lay in unsupported_layers:
        if lay is not None:
            with pytest.warns(Warning):
                _ = uft.generate_k_lip_model(
                    Sequential,
                    {"layers": [lay]},
                    input_shape=None,
                    k=None,
                )


def test_vanilla_export_with_named_layers():
    input_shape = uft.to_framework_channel((3, 20, 20))
    feat = uft.generate_k_lip_model(
        Sequential, {"layers": sequential_layers(input_shape)}, input_shape=input_shape
    )
    cl = uft.get_instance_framework(
        FrobeniusLinear, {"in_features": 10, "out_features": 1}
    )
    model = uft.build_named_sequential(
        Sequential, OrderedDict([("features", feat), ("classifier", cl)])
    )
    x = np.random.normal(size=(1,) + input_shape)
    x = uft.to_tensor(x)
    _ = model(x)
    names = [name for name, _ in uft.get_named_children(model.vanilla_export())]
    assert names == ["features", "classifier"]
