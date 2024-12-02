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
import os
import pprint
import pytest
import numpy as np
from .test_layers import linear_generator, build_kernel

from . import utils_framework as uft

from tests.utils_framework import (
    Sequential,
    tModel,
)

from tests.utils_framework import (
    SpectralLinear,
    SpectralConv2d,
    SpectralConvTranspose2d,
    FrobeniusLinear,
    FrobeniusConv2d,
    ScaledL2NormPool2d,
)
from tests.utils_framework import Flatten, tLinear, tInput

np.random.seed(42)
pp = pprint.PrettyPrinter(indent=4)


def sequential_layers(input_shape):
    """Return list of layers for a Sequential model"""
    in_ch = input_shape[0]
    input_shape = uft.to_framework_channel(input_shape)
    return [
        uft.get_instance_framework(tInput, {"shape": input_shape}),
        uft.get_instance_framework(
            SpectralConv2d,
            {
                "in_channels": in_ch,
                "out_channels": 2,
                "kernel_size": (3, 3),
                "padding": 1,
            },
        ),
        uft.get_instance_framework(ScaledL2NormPool2d, {"kernel_size": (2, 2)}),
        uft.get_instance_framework(
            FrobeniusConv2d,
            {"in_channels": 2, "out_channels": 2, "kernel_size": (3, 3), "padding": 1},
        ),
        uft.get_instance_framework(
            SpectralConvTranspose2d,
            {"in_channels": 2, "out_channels": 5, "kernel_size": (3, 3), "padding": 1},
        ),
        uft.get_instance_framework(Flatten, {}),
        uft.get_instance_framework(tLinear, {"in_features": 80, "out_features": 4}),
        uft.get_instance_framework(
            SpectralLinear, {"in_features": 4, "out_features": 4}
        ),
        uft.get_instance_framework(
            FrobeniusLinear, {"in_features": 4, "out_features": 2}
        ),
    ]


def get_functional_tensors(input_shape):
    in_ch = input_shape[0]
    input_shape = uft.to_framework_channel(input_shape)
    dict_functional_tensors = {}
    dict_functional_tensors["inputs"] = uft.get_instance_framework(
        tInput, {"shape": input_shape}
    )
    dict_functional_tensors["conv1"] = uft.get_instance_framework(
        SpectralConv2d,
        {
            "in_channels": in_ch,
            "out_channels": 2,
            "kernel_size": (3, 3),
            "k_coef_lip": 2.0,
            "padding": 1,
        },
    )
    dict_functional_tensors["pool1"] = uft.get_instance_framework(
        ScaledL2NormPool2d, {"kernel_size": (2, 2), "k_coef_lip": 2.0}
    )
    dict_functional_tensors["conv2"] = uft.get_instance_framework(
        FrobeniusConv2d,
        {
            "in_channels": 2,
            "out_channels": 2,
            "kernel_size": (3, 3),
            "k_coef_lip": 2.0,
            "padding": 1,
        },
    )
    dict_functional_tensors["convt2"] = uft.get_instance_framework(
        SpectralConvTranspose2d,
        {"in_channels": 2, "out_channels": 5, "kernel_size": (3, 3), "padding": 1},
    )
    dict_functional_tensors["flatten"] = uft.get_instance_framework(Flatten, {})
    dict_functional_tensors["dense1"] = uft.get_instance_framework(
        tLinear, {"in_features": 80, "out_features": 4}
    )
    dict_functional_tensors["dense2"] = uft.get_instance_framework(
        SpectralLinear, {"in_features": 4, "out_features": 4}
    )
    dict_functional_tensors["dense3"] = uft.get_instance_framework(
        FrobeniusLinear, {"in_features": 4, "out_features": 2}
    )
    return dict_functional_tensors


def functional_input_output_tensors(dict_functional_tensors, x):
    """Return input and output tensor of a Functional (hard-coded) model"""
    if dict_functional_tensors["inputs"] is None:
        inputs = x
    else:
        inputs = dict_functional_tensors["inputs"]
    x = dict_functional_tensors["conv1"](inputs)
    x = dict_functional_tensors["pool1"](x)
    x = dict_functional_tensors["conv2"](x)
    x = dict_functional_tensors["convt2"](x)
    x = dict_functional_tensors["flatten"](x)
    x = dict_functional_tensors["dense1"](x)
    x = dict_functional_tensors["dense2"](x)
    outputs = dict_functional_tensors["dense3"](x)
    if dict_functional_tensors["inputs"] is None:
        return outputs
    else:
        return inputs, outputs
    # return x


def get_model(model_type, layer_params, input_shape, k_coef_lip):
    if model_type == tModel:
        return uft.get_functional_model(
            tModel,
            layer_params["dict_tensors"],
            layer_params["functional_input_output_tensors"],
        )
    else:
        return uft.generate_k_lip_model(
            model_type, layer_params, input_shape=input_shape, k=k_coef_lip
        )


@pytest.mark.skipif(
    hasattr(SpectralConvTranspose2d, "unavailable_class"),
    reason="SpectralConvTranspose2d not available",
)
@pytest.mark.parametrize(
    "model_type, params_type, param_fct, dict_other_params, k_coef_lip, input_shape",
    [
        (Sequential, "layers", sequential_layers, {}, 5.0, (3, 8, 8)),
        (
            tModel,
            "dict_tensors",
            get_functional_tensors,
            {
                "functional_input_output_tensors": functional_input_output_tensors,
            },
            5.0,
            (3, 8, 8),
        ),
    ],
)
def test_model(
    model_type, params_type, param_fct, dict_other_params, k_coef_lip, input_shape
):
    batch_size = 250
    epochs = 1
    steps_per_epoch = 125
    k_lip_data = 2.0

    # clear session to avoid side effects from previous train
    uft.init_session()  # K.clear_session()
    np.random.seed(42)
    input_shape_CHW = input_shape
    input_shape = uft.to_framework_channel(input_shape)
    layer_params = {params_type: param_fct(input_shape_CHW)}
    layer_params.update(dict_other_params)
    model = get_model(model_type, layer_params, input_shape, k_coef_lip)

    # create the model, defin opt, and compile it
    optimizer = uft.get_instance_framework(
        uft.Adam, inst_params={"lr": 0.001, "model": model}
    )

    loss_fn, optimizer, metrics = uft.compile_model(
        model,
        optimizer=optimizer,
        loss=uft.MeanSquaredError(),
        metrics=[uft.metric_mse()],
    )
    # create the synthetic data generator
    output_shape = uft.compute_output_shape(input_shape, model)
    # output_shape = model.uft.compute_output_shape((batch_size,) + input_shape)[1:]
    kernel = build_kernel(input_shape, output_shape, k_lip_data)
    # define logging features
    logdir = os.path.join(
        "logs", uft.LIP_LAYERS, "condense_test"
    )  # , datetime.now().strftime("%Y_%m_%d-%H_%M_%S")))
    os.makedirs(logdir, exist_ok=True)
    callback_list = []  # callbacks.TensorBoard(logdir)]
    # train model

    traind_ds = linear_generator(batch_size, input_shape, kernel)
    uft.train(
        traind_ds,
        model,
        loss_fn,
        optimizer,
        epochs,
        batch_size,
        steps_per_epoch=steps_per_epoch,
        callbacks=callback_list,
    )
    # the seed is set to compare all models with the same data
    np.random.seed(42)
    # get original results
    test_dl = linear_generator(batch_size, input_shape, kernel)
    loss, mse = uft.run_test(model, test_dl, loss_fn, metrics, steps=10)
    # generate vanilla
    if uft.vanilla_require_a_copy():
        layer_params = {params_type: param_fct(input_shape_CHW)}
        layer_params.update(dict_other_params)
        model2 = get_model(model_type, layer_params, input_shape, k_coef_lip)
        uft.copy_model_parameters(model, model2)
        vanilla_model = uft.vanillaModel(model2)
    else:
        vanilla_model = uft.vanillaModel(model)
    # vanilla_model = model.vanilla_export()
    loss_fn, optimizer, metrics = uft.compile_model(
        vanilla_model,
        optimizer=optimizer,
        loss=uft.MeanSquaredError(),
        metrics=[uft.metric_mse()],
    )
    np.random.seed(42)
    # evaluate vanilla
    test_dl = linear_generator(batch_size, input_shape, kernel)
    loss2, mse2 = uft.run_test(model, test_dl, loss_fn, metrics, steps=10)
    np.random.seed(42)
    # check if original has changed
    test_dl = linear_generator(batch_size, input_shape, kernel)
    vanilla_loss, vanilla_mse = uft.run_test(
        vanilla_model, test_dl, loss_fn, metrics, steps=10
    )

    np.testing.assert_almost_equal(
        mse,
        vanilla_mse,
        3,
        "the exported vanilla model must have same behaviour as original",
    )
    np.testing.assert_equal(
        mse, mse2, "exporting a model must not change original model"
    )
    # add one epoch to orginal

    traind_ds = linear_generator(batch_size, input_shape, kernel)
    uft.train(
        traind_ds,
        model,
        loss_fn,
        optimizer,
        epochs,
        batch_size,
        steps_per_epoch=steps_per_epoch,
        callbacks=callback_list,
    )
    np.random.seed(42)
    test_dl = linear_generator(batch_size, input_shape, kernel)
    loss3, mse3 = uft.run_test(model, test_dl, loss_fn, metrics, steps=10)
    # check if vanilla has changed
    np.random.seed(42)
    test_dl = linear_generator(batch_size, input_shape, kernel)
    vanilla_loss2, vanilla_mse2 = uft.run_test(
        vanilla_model, test_dl, loss_fn, metrics, steps=10
    )
    np.testing.assert_equal(
        vanilla_mse,
        vanilla_mse2,
        "exported model must be completely independent from original",
    )

    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(
            mse,
            mse3,
            4,
            "all tests passe but integrity check failed: the test cannot conclude that "
            + "vanilla_export create a distinct model",
        )
