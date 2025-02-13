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
import pytest
import numpy as np
import os

from . import utils_framework as uft

from .utils_framework import (
    tLinear,
    CategoricalProvableRobustAccuracy,
    BinaryProvableRobustAccuracy,
    CategoricalProvableAvgRobustness,
    BinaryProvableAvgRobustness,
)


def one_hot_data(x, n_class):
    """Return a Framework float32 tensor of shape [N, n_class]
    from a list/np.array of shape [N]"""
    return np.eye(n_class)[x]


@pytest.mark.parametrize(
    "nb_class, loss, loss_params, nb_classes",
    [
        (
            10,
            CategoricalProvableRobustAccuracy,
            {"epsilon": 1, "lip_const": 1.0, "disjoint_neurons": False},
            10,
        ),
        (
            10,
            BinaryProvableRobustAccuracy,
            {
                "epsilon": 1,
                "lip_const": 1.0,
            },
            10,
        ),
        (
            10,
            CategoricalProvableAvgRobustness,
            {"lip_const": 2.0, "disjoint_neurons": False, "negative_robustness": True},
            10,
        ),
        (
            10,
            BinaryProvableAvgRobustness,
            {"lip_const": 2.0, "negative_robustness": True},
            10,
        ),
    ],
)
def test_serialization(nb_class, loss, loss_params, nb_classes):
    if hasattr(loss, "unavailable_class"):
        pytest.skip(f"{loss} not implemented")

    n = 255
    x = np.random.uniform(size=(n, 42))
    y = one_hot_data(np.random.randint(nb_classes, size=n), nb_classes)
    x, y = uft.to_tensor(x), uft.to_tensor(y)
    m = uft.generate_k_lip_model(
        tLinear,
        layer_params={"in_features": 42, "out_features": nb_class},
        input_shape=(42,),
    )

    loss_fn, optimizer, metrics = uft.compile_model(
        m,
        optimizer=uft.get_instance_framework(
            uft.SGD, inst_params={"lr": 0.001, "model": m}
        ),
        loss=uft.get_instance_framework(loss, inst_params=loss_params),
    )

    l1 = m.evaluate(x, y)
    name = loss.__class__.__name__
    path = os.path.join("logs", "metrics", name)

    uft.save_model(m, path)
    m2 = uft.load_model(
        path,
        compile=True,
        layer_type=tLinear,
        layer_params={"in_features": 42, "out_features": nb_class},
        input_shape=(42,),
        k=1,
    )
    l2 = m2.evaluate(x, y)
    np.testing.assert_equal(
        l1,
        l2,
        err_msg=f"serialization changed loss value for {loss}",
    )
    return


def build_test_provable_vs_adjusted():
    test_data = []
    if not hasattr(CategoricalProvableAvgRobustness, "unavailable_class"):
        for lip_cst in [1, 0.5, 2.0]:
            for disjoint in [True, False]:
                test_data.append(
                    (
                        CategoricalProvableAvgRobustness,
                        {
                            "lip_const": lip_cst,
                            "disjoint_neurons": disjoint,
                            "negative_robustness": True,
                        },
                        10,
                    )
                )

    if not hasattr(BinaryProvableAvgRobustness, "unavailable_class"):
        for lip_cst in [1.0, 0.5, 2.0]:
            test_data.append(
                (
                    BinaryProvableAvgRobustness,
                    {"lip_const": lip_cst, "negative_robustness": True},
                    1,
                )
            )
    return test_data


@pytest.mark.parametrize(
    "loss, loss_params, nb_class", build_test_provable_vs_adjusted()
)
def test_provable_vs_adjusted(loss, loss_params, nb_class):
    n = 500
    x = np.random.uniform(size=(n, nb_class))
    if nb_class > 1:
        y = one_hot_data(np.random.randint(nb_class, size=n), nb_class)
    else:
        y = np.random.randint(2, size=n)
    x, y = uft.to_tensor(x), uft.to_tensor(y)

    pr = uft.get_instance_framework(loss, inst_params=loss_params)
    if pr is None:
        pytest.skip(f"{loss} not implemented")
    other_param = loss_params.copy()
    other_param["negative_robustness"] = not loss_params["negative_robustness"]
    ar = uft.get_instance_framework(loss, inst_params=other_param)
    l1 = pr(y, y).numpy()
    l2 = ar(y, y).numpy()
    np.testing.assert_allclose(
        l1,
        l2,
        1e-4,
        err_msg=f"{loss} provable and adjusted "
        "robustness must give same "
        "values when y_true==y_pred",
    )

    l1 = pr(y, x).numpy()
    l2 = ar(y, x).numpy()
    diff = np.min(np.abs(l1 - l2))
    assert (
        diff > 1e-4
    ), f"{loss} provable and adjusted robustness must give \
        different values when y_true!=y_pred"


@pytest.mark.parametrize(
    "loss, loss_params, nb_class",
    [
        (
            CategoricalProvableAvgRobustness,
            {
                "lip_const": 1.0,
                "disjoint_neurons": True,
                "negative_robustness": True,
            },
            10,
        ),
        (
            BinaryProvableAvgRobustness,
            {"lip_const": 1.0, "negative_robustness": True},
            1,
        ),
    ],
)
def test_data_format(loss, loss_params, nb_class):
    n = 500

    if hasattr(loss, "unavailable_class"):
        pytest.skip(f"{loss} not implemented")
    pr = uft.get_instance_framework(loss, inst_params=loss_params)

    other_param = loss_params.copy()
    other_param["negative_robustness"] = not loss_params["negative_robustness"]
    ar = uft.get_instance_framework(loss, inst_params=other_param)

    for metric in [pr, ar]:
        x = np.random.uniform(size=(n, nb_class))
        if nb_class > 1:
            y = one_hot_data(np.random.randint(nb_class, size=n), nb_class)
        else:
            y = np.random.randint(2, size=n)
        metrics_values = []
        for neg_val in [0.0, -1.0]:
            y_pred_case = x
            y_true_case = np.where(y > 0, 1.0, neg_val)
            y_pred_case, y_true_case = (
                uft.to_tensor(y_pred_case),
                uft.to_tensor(y_true_case),
            )
            metrics_values.append(metric(y_true_case, y_pred_case).numpy())
        assert all(
            [m == metrics_values[0] for m in metrics_values]
        ), f"{loss} changing the data format must not change the metric value"


@pytest.mark.parametrize(
    "loss, loss_params, nb_class",
    [
        (
            CategoricalProvableAvgRobustness,
            {
                "lip_const": 1.0,
                "disjoint_neurons": False,
                "negative_robustness": True,
            },
            10,
        ),
        (
            CategoricalProvableAvgRobustness,
            {
                "lip_const": 1.0,
                "disjoint_neurons": False,
                "negative_robustness": False,
            },
            10,
        ),
    ],
)
def test_disjoint_neurons(loss, loss_params, nb_class):
    n = 500

    if hasattr(loss, "unavailable_class"):
        pytest.skip(f"{loss} not implemented")

    pr = uft.get_instance_framework(loss, inst_params=loss_params)
    # if pr is None:
    #     pytest.skip(f"{loss} not implemented")
    other_param = loss_params.copy()
    other_param["disjoint_neurons"] = True
    pdr = uft.get_instance_framework(loss, inst_params=other_param)

    x = np.random.uniform(size=(n, nb_class))
    if nb_class > 1:
        y = one_hot_data(np.random.randint(nb_class, size=n), nb_class)
    else:
        y = np.random.randint(2, size=n)
    metrics_values = []
    for neg_val in [0.0, -1.0]:
        y_pred_case = x
        y_true_case = np.where(y > 0, 1.0, neg_val)
        y_pred_case, y_true_case = (
            uft.to_tensor(y_pred_case),
            uft.to_tensor(y_true_case),
        )
        metrics_values.append(pr(y_true_case, y_pred_case).numpy())
        metrics_values.append(
            pdr(y_true_case, y_pred_case * (np.sqrt(2) / 2.0)).numpy()
        )
    np.testing.assert_allclose(
        metrics_values[0],
        metrics_values,
        4,
        err_msg=f"{loss} does not account disjoint_neuron properly",
    )


y_pred1 = [
    [1.0, 0.0, 0.0],  # good class & over 0.25
    [0.1, 0.0, 0.0],  # good class & below 0.25
    [0.0, 0.0, 1.0],  # wrong class & over 0.25
    [0.0, 0.0, 0.1],  # wrong class & below 0.25
]
y_true1 = [
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
]
y_pred2 = [
    1.0,  # good class & over 0.25
    0.1,  # good class & below 0.25
    1.0,  # wrong class & over 0.25
    0.1,  # wrong class & below 0.25
    -1.0,
    -0.1,
    -1.0,
    -0.1,
]
y_true2 = [1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0]


@pytest.mark.parametrize(
    "loss, loss_params, y_pred, y_true, expected_value",
    [
        (
            CategoricalProvableRobustAccuracy,
            {
                "epsilon": 0.25,
                "lip_const": 1.0,
                "disjoint_neurons": False,
            },
            y_pred1,
            y_true1,
            0.25,
        ),
        (
            CategoricalProvableAvgRobustness,
            {
                "lip_const": 1.0,
                "disjoint_neurons": False,
            },
            y_pred1,
            y_true1,
            0.25 * 1.1 / np.sqrt(2),
        ),
        (
            BinaryProvableRobustAccuracy,
            {
                "epsilon": 0.25,
                "lip_const": 1.0,
            },
            y_pred2,
            y_true2,
            0.25,
        ),
        (
            BinaryProvableAvgRobustness,
            {
                "lip_const": 1.0,
                "negative_robustness": False,
            },
            y_pred2,
            y_true2,
            0.125 * (1.1 * 2),
        ),
    ],
)
def test_hardcoded_values(loss, loss_params, y_pred, y_true, expected_value):
    if hasattr(loss, "unavailable_class"):
        pytest.skip(f"{loss} not implemented")
    pr = uft.get_instance_framework(loss, inst_params=loss_params)
    y_pred, y_true = uft.to_tensor(y_pred), uft.to_tensor(y_true)
    val = pr(y_true, y_pred).numpy()
    np.testing.assert_allclose(
        val,
        expected_value,
        5,
        err_msg=f"{loss}  did not get the expected result with hardcoded values ",
    )
