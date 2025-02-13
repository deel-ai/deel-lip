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
import os
import numpy as np

from . import utils_framework as uft

from .utils_framework import (
    tLinear,
    Loss,
    KRLoss,
    HingeMarginLoss,
    HKRLoss,
    KRMulticlassLoss,
    HingeMulticlassLoss,
    HKRMulticlassLoss,
    SoftHKRMulticlassLoss,
    MultiMarginLoss,
    TauCategoricalCrossentropyLoss,
    TauSparseCategoricalCrossentropyLoss,
    TauBinaryCrossentropyLoss,
    CategoricalHingeLoss,
    process_labels_for_multi_gpu,
)


# a tester:
# - un cas hardcodé
# - les dtypes pour y_true
# - shape y_pred
# - labels en [0,1] et en [-1,1]
# - liens entre les losses


def get_gaussian_data(n=500, mean1=1.0, mean2=-1.0):
    x1 = np.random.normal(size=(n, 1), loc=mean1, scale=0.1)
    x2 = np.random.normal(size=(n, 1), loc=mean2, scale=0.1)
    y_pred = np.concatenate([x1, x2], axis=0)
    y_true = np.concatenate([np.ones((n, 1)), np.zeros((n, 1))], axis=0)
    return y_pred, y_true


def check_serialization(nb_class, loss):
    layer_type = tLinear
    layer_params = {"in_features": 10, "out_features": nb_class}
    m = uft.generate_k_lip_model(layer_type, layer_params, input_shape=(10,), k=1)
    assert m is not None
    loss, optimizer, _ = uft.compile_model(
        m,
        optimizer=uft.get_instance_framework(uft.SGD, inst_params={"model": m}),
        loss=loss,
    )
    name = loss.__class__.__name__ if isinstance(loss, Loss) else loss.__name__
    path = os.path.join("logs", "losses", name)

    uft.save_model(m, path)
    m2 = uft.load_model(
        path,
        compile=True,
        layer_type=layer_type,
        layer_params=layer_params,
        input_shape=(10,),
        k=1,
    )
    x = np.random.uniform(size=(255, 10))
    x = uft.to_tensor(x)
    m2(x)


def binary_data(x):
    """Return a Framework float32 tensor of shape [N, 1]
    from a list/np.array of shape [N]"""
    return np.expand_dims(np.array(x, dtype=np.float32), axis=-1)


def one_hot_data(x, n_class):
    """Return a Framework float32 tensor of shape [N, n_class]
    from a list/np.array of shape [N]"""
    return np.eye(n_class)[x]


# global test values
y_true1 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
y_pred1 = [0.5, 1.5, -0.5, -0.5, -1.5, 0.5]

y_true1b = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
y_pred1b = [0.5, 1.1, -0.1, 0.7, -1.3, -0.4, 0.2, -0.9]
y_true2 = one_hot_data([0, 0, 0, 1, 1, 2], 3)
y_pred2 = np.float32(
    [
        [2, 0.2, -0.5],
        [-1, -1.2, 0.3],
        [0.8, 2, 0],
        [0, 1, -0.5],
        [2.4, -0.4, -1.1],
        [-0.1, -1.7, 0.6],
    ]
)

y_true3 = one_hot_data([0, 0, 0, 1, 1, 1, 2, 2], 3)
y_pred3 = np.array(
    [
        [1.5, 0.2, -0.5],
        [1, -1.2, 0.3],
        [0.8, 2, -2],
        [0, 1, -0.5],
        [-0.8, 2, 0],
        [2.4, -0.4, -1.1],
        [-0.1, -1.7, 0.6],
        [1.2, 1.3, 2.5],
    ],
    dtype=np.float32,
)
results_tau_cat = np.float64(
    [
        0.044275,
        0.115109,
        1.243572,
        0.084923,
        0.010887,
        2.802300,
        0.114224,
        0.076357,
    ]
)
y_predgaussian, y_truegaussian = get_gaussian_data(20000)
n_class = 10
n_items = 10000
y_truelabels = np.random.randint(0, n_class, n_items)
y_trueonehot = one_hot_data(np.random.randint(0, n_class, n_items), n_class)
y_predonehot = np.random.normal(size=(n_items, n_class))
# Large random data of size [10000, 1]
num_items4 = 10000
y_true4 = np.random.randint(2, size=num_items4)
y_pred4 = np.random.normal(size=(num_items4,))


# TODO add  HKR tests
@pytest.mark.parametrize(
    "loss_instance, loss_params, y_true_np, y_pred_np, expected_loss,rtol",
    [
        (
            KRLoss,
            {},
            binary_data(y_true1),
            binary_data(y_pred1),
            1.0,
            1e-7,
        ),
        (KRLoss, {}, y_truegaussian, y_predgaussian, 2.0, 1e-3),
        (
            HingeMarginLoss,
            {"min_margin": 2.0},
            binary_data(y_true1),
            binary_data(y_pred1),
            4 / 6,
            1e-7,
        ),
        (
            KRMulticlassLoss,
            {"reduction": "auto"},
            y_true2,
            y_pred2,
            np.float32(761.0 / 1800.0),
            1e-7,
        ),
        (
            HKRMulticlassLoss,
            {"alpha": 5.0, "min_margin": 2.0},
            y_true2,
            y_pred2,
            np.float32(1071.0 / 200.0) * uft.scaleAlpha(5.0),
            1e-7,
        ),
        (
            SoftHKRMulticlassLoss,
            {
                "alpha": 5.0,
                "min_margin": 0.2,
            },  # Warning alpha replaced by alpha/(1+alpha)
            y_true2,
            y_pred2,
            np.float32(1.0897621 * uft.scaleDivAlpha(5.0)),
            1e-5,
        ),
        (
            TauBinaryCrossentropyLoss,
            {"tau": 2.0},
            binary_data(y_true1),
            binary_data(y_pred1),
            0.279185,
            1e-6,
        ),
    ],
)
def test_loss_generic_value(
    loss_instance, loss_params, y_true_np, y_pred_np, expected_loss, rtol
):
    if hasattr(loss_instance, "unavailable_class"):
        pytest.skip(f"{loss_instance} not implemented")
    loss = uft.get_instance_framework(loss_instance, inst_params=loss_params)
    if loss is None:
        pytest.skip(f"{loss_instance}  with params {loss_params} not implemented")
    y_true, y_pred = uft.to_tensor(y_true_np), uft.to_tensor(y_pred_np)

    loss_val = uft.compute_loss(loss, y_pred, y_true).numpy()
    np.testing.assert_allclose(
        loss_val,
        np.float32(expected_loss),
        rtol=rtol,
        err_msg=f"{loss_instance} loss must be equal to {expected_loss}",
    )


@pytest.mark.parametrize(
    "loss_instance, loss_params, y_true_np, y_pred_np, test_minus1",
    [
        (KRLoss, {}, y_truegaussian, y_predgaussian, True),
        (
            HingeMarginLoss,
            {"min_margin": 2.0},
            binary_data(y_true1),
            binary_data(y_pred1),
            True,
        ),
        (
            KRMulticlassLoss,
            {"reduction": "auto"},
            y_trueonehot,
            y_predonehot,
            True,
        ),  # OK (-1,1) but no more one hot
        (
            HingeMulticlassLoss,
            {"min_margin": 2.0},
            y_trueonehot,
            y_predonehot,
            True,
        ),  # OK (-1,1) but no more one hot
        (
            HKRMulticlassLoss,
            {"alpha": 5.0, "min_margin": 2.0},
            y_trueonehot,
            y_predonehot,
            True,
        ),  # OK (-1,1) but no more one hot
        (
            MultiMarginLoss,
            {"min_margin": 1.0},
            y_trueonehot,
            y_predonehot,
            True,
        ),  # OK (-1,1) but no more one hot
        (
            CategoricalHingeLoss,
            {"min_margin": 1.0},
            y_trueonehot,
            y_predonehot,
            True,
        ),  # OK (-1,1) but no more one hot
        (
            TauCategoricalCrossentropyLoss,
            {"tau": 1.0},
            y_trueonehot,
            y_predonehot,
            False,
        ),
        (
            TauSparseCategoricalCrossentropyLoss,
            {"tau": 1.0},
            y_truelabels,
            y_predonehot,
            False,
        ),
        (
            TauBinaryCrossentropyLoss,
            {"tau": 0.5},
            binary_data(y_true1),
            binary_data(y_pred1),
            True,
        ),
    ],
)
def test_loss_generic_equal(
    loss_instance, loss_params, y_true_np, y_pred_np, test_minus1
):
    if hasattr(loss_instance, "unavailable_class"):
        pytest.skip(f"{loss_instance} not implemented")
    loss = uft.get_instance_framework(loss_instance, inst_params=loss_params)
    if loss is None:
        pytest.skip(f"{loss_instance} with params {loss_params} not implemented")
    y_true, y_pred = uft.to_tensor(y_true_np), uft.to_tensor(y_pred_np)
    loss_val = uft.compute_loss(loss, y_pred, y_true).numpy()
    y_true = uft.to_tensor(y_true_np, dtype=uft.type_int32)
    loss_val_2 = uft.compute_loss(loss, y_pred, y_true).numpy()
    np.testing.assert_equal(
        loss_val_2,
        loss_val,
        err_msg=f"{loss_instance} test failed when y_true has dtype int32",
    )
    if test_minus1:
        y_true2_np = np.where(y_true_np == 1.0, 1.0, -1.0)
        y_true2 = uft.to_tensor(y_true2_np)
        loss_val_3 = uft.compute_loss(loss, y_pred, y_true2).numpy()
        np.testing.assert_equal(
            loss_val_3,
            loss_val,
            err_msg="{loss_instance} test failed when labels are in (1, -1)",
        )

    check_serialization(1, loss)


def test_hkr_loss():
    # assuming KRLoss and hinge have been properly tested
    loss_instance = HKRLoss
    if hasattr(loss_instance, "unavailable_class"):
        pytest.skip(f"{loss_instance} not implemented")
    loss = uft.get_instance_framework(
        loss_instance,
        inst_params={"alpha": 0.5, "min_margin": 2.0},
    )
    if loss is None:
        pytest.skip(f"{loss_instance} with params not implemented")
    check_serialization(1, loss)


def test_softhkrmulticlass_loss():
    global y_true2, y_pred2, y_trueonehot, y_predonehot
    loss_instance = SoftHKRMulticlassLoss
    if hasattr(loss_instance, "unavailable_class"):
        pytest.skip(f"{loss_instance} not implemented")
    y_true_np, y_pred_np = y_true2, y_pred2
    expected_loss = np.float32(1.0897621) * uft.scaleDivAlpha(
        5.0
    )  # warning alpha scaled to be in [0,1]
    rtol = 1e-5
    loss = uft.get_instance_framework(
        loss_instance, inst_params={"alpha": 5.0, "min_margin": 0.2}
    )
    if loss is None:
        pytest.skip(f"{loss_instance}   with params  not implemented")
    y_true, y_pred = uft.to_tensor(y_true_np), uft.to_tensor(y_pred_np)
    loss_val = uft.compute_loss(loss, y_pred, y_true).numpy()
    np.testing.assert_allclose(loss_val, np.float32(expected_loss), rtol=rtol)

    # moving mean should change and thus loss value
    loss_val_bis = uft.compute_loss(loss, y_pred, y_true).numpy()
    np.testing.assert_allclose(
        loss_val_bis, np.float32(1.0834466) * uft.scaleDivAlpha(5.0), rtol=rtol
    )

    y_true_np, y_pred_np = y_trueonehot, y_predonehot
    loss = uft.get_instance_framework(
        loss_instance, inst_params={"alpha": 5.0, "min_margin": 0.2}
    )
    y_true, y_pred = uft.to_tensor(y_true_np), uft.to_tensor(y_pred_np)
    loss_val = uft.compute_loss(loss, y_pred, y_true).numpy()

    # equality require a new instance
    loss2 = uft.get_instance_framework(
        loss_instance, inst_params={"alpha": 5.0, "min_margin": 0.2}
    )
    y_true = uft.to_tensor(y_true_np, dtype=uft.type_int32)
    loss_val_2 = uft.compute_loss(loss2, y_pred, y_true).numpy()
    np.testing.assert_equal(
        loss_val_2,
        loss_val,
        err_msg=f"{loss_instance} test failed when y_true has dtype int32",
    )

    loss3 = uft.get_instance_framework(
        loss_instance, inst_params={"alpha": 5.0, "min_margin": 0.2}
    )
    y_true2_np = np.where(y_true_np == 1.0, 1.0, -1.0)
    y_true2 = uft.to_tensor(y_true2_np)
    loss_val_3 = uft.compute_loss(loss3, y_pred, y_true2).numpy()
    np.testing.assert_equal(
        loss_val_3,
        loss_val,
        err_msg="{loss_instance} test failed when labels are in (1, -1)",
    )

    check_serialization(1, loss)


@pytest.mark.parametrize(
    "loss_instance, loss_params, y_true_np, y_pred_np, expected_loss, rtol",
    [
        (
            KRLoss,
            {"reduction": "none"},
            binary_data(y_true1b),
            binary_data(y_pred1b),
            [1.0, 2.2, -0.2, 1.4, 2.6, 0.8, -0.4, 1.8],
            1e-7,
        ),
        (
            HingeMarginLoss,
            {"min_margin": 0.7 * 2.0, "reduction": "none"},
            binary_data(y_true1b),
            binary_data(y_pred1b),
            [0.2, 0, 0.8, 0, 0, 0.3, 0.9, 0],
            1e-7,
        ),
        (
            HKRLoss,
            {
                "alpha": 2.5,
                "min_margin": 2.0,
                "reduction": "none",
            },
            binary_data(y_true1b),
            binary_data(y_pred1b),
            np.float64([0.25, -2.2, 2.95, -0.65, -2.6, 0.7, 3.4, -1.55])
            * uft.scaleAlpha(2.5),
            1e-7,
        ),
        (
            TauBinaryCrossentropyLoss,
            {"tau": 0.5, "reduction": "none"},
            binary_data(y_true1b),
            binary_data(y_pred1b),
            [1.15188, 0.91098, 1.43692, 1.06676, 0.84011, 1.19628, 1.48879, 0.98650],
            1e-7,
        ),
        (
            KRMulticlassLoss,
            {"reduction": "none"},
            y_true3,
            y_pred3,
            np.float64([326, 314, 120, 250, 496, -258, 396, 450]) / 225,
            1e-7,
        ),  # OK (-1,1) but no more one hot
        (
            HingeMulticlassLoss,
            {"min_margin": 2.0, "reduction": "none"},
            y_true3,
            y_pred3,
            np.float64([17, 13, 34, 15, 12, 62, 17, 45]) / 30,
            1e-7,
        ),  # OK (-1,1) but no more one hot
        (
            MultiMarginLoss,
            {"min_margin": 0.7, "reduction": "none"},
            y_true3,
            y_pred3,
            np.float64([0, 0, 19, 0, 0, 35, 0, 0]) / 30,
            1e-7,
        ),  # OK (-1,1) but no more one hot
        (
            HKRMulticlassLoss,
            {"alpha": 2.5, "min_margin": 1.0, "reduction": "none"},
            y_true3,
            y_pred3,
            np.float64([-779, -656, 1395, -625, -1609, 4557, -1284, 825])
            * uft.scaleAlpha(2.5)
            / 900,
            1e-7,
        ),  # OK (-1,1) but no more one hot
        (
            CategoricalHingeLoss,
            {"min_margin": 1.1, "reduction": "none"},
            y_true3,
            y_pred3,
            np.float64([0, 0.4, 2.3, 0.1, 0, 3.9, 0.4, 0]),
            1e-7,
        ),  # OK (-1,1) but no more one hot
        (
            TauCategoricalCrossentropyLoss,
            {"tau": 2.0, "reduction": "none"},
            y_true3,
            y_pred3,
            results_tau_cat,
            1e-7,
        ),
        (
            TauSparseCategoricalCrossentropyLoss,
            {"tau": 2.0, "reduction": "none"},
            np.argmax(y_true3, axis=-1),
            y_pred3,
            results_tau_cat,
            1e-7,
        ),
        (
            SoftHKRMulticlassLoss,
            {"alpha": 5.0, "min_margin": 0.2, "reduction": "none"},
            y_true3,
            y_pred3,
            np.float64(
                [
                    -0.10878129,
                    0.12582462,
                    2.2788424,
                    -0.17646402,
                    -0.38242298,
                    3.4523692,
                    -0.19672059,
                    1.1028761,
                ]
            )
            * uft.scaleDivAlpha(5.0),
            1e-7,
        ),
    ],
)
def test_no_reduction_loss_generic(
    loss_instance, loss_params, y_true_np, y_pred_np, expected_loss, rtol
):
    """
    Assert binary losses without reduction. Three losses are tested on hardcoded
    y_true/y_pred of shape [8 elements, 1]: KRLoss, HingeMarginLoss and HKRLoss.
    """
    if hasattr(loss_instance, "unavailable_class"):
        pytest.skip(f"{loss_instance} not implemented")
    loss = uft.get_instance_framework(loss_instance, inst_params=loss_params)
    if loss is None:
        pytest.skip(f"{loss_instance}   with params {loss_params} not implemented")
    y_true, y_pred = uft.to_tensor(y_true_np), uft.to_tensor(y_pred_np)
    loss_val = uft.to_numpy(uft.compute_loss(loss, y_pred, y_true))
    loss_val = np.squeeze(loss_val)
    np.testing.assert_allclose(
        loss_val,
        expected_loss,
        rtol=rtol,
        atol=5e-6,
        err_msg=f"Loss {loss} failed",
    )


segments4 = [
    0,
    num_items4 // 2,
    num_items4 // 2 + num_items4 // 4,
    num_items4 // 2 + num_items4 // 4 + num_items4 // 6,
    None,
]
# segments4 = [0,None]


@pytest.mark.skipif(
    hasattr(process_labels_for_multi_gpu, "unavailable_class"),
    reason="process_labels_for_multi_gpu not available",
)
@pytest.mark.parametrize(
    "loss_instance, loss_params, y_true_np, y_pred_np, segments, expected_loss, rtol",
    [
        (
            KRLoss,
            {"multi_gpu": True, "reduction": "sum"},
            binary_data(y_true1b),
            binary_data(y_pred1b),
            [0, 3, 7, None],
            9.2,
            1e-7,
        ),
        (
            HingeMarginLoss,
            {"min_margin": 0.7 * 2.0, "reduction": "sum"},
            binary_data(y_true1b),
            binary_data(y_pred1b),
            [0, 3, 7, None],
            2.2,
            1e-7,
        ),
        (
            HKRLoss,
            {
                "alpha": 2.5,
                "min_margin": 2.0,
                "reduction": "sum",
                "multi_gpu": True,
            },
            binary_data(y_true1b),
            binary_data(y_pred1b),
            [0, 3, 7, None],
            0.3 * uft.scaleAlpha(2.5),
            1e-7,
        ),
        (
            TauBinaryCrossentropyLoss,
            {"tau": 1.5, "reduction": "sum"},
            binary_data(y_true1b),
            binary_data(y_pred1b),
            [0, 3, 7, None],
            2.19262,
            1e-7,
        ),
        (
            KRLoss,
            {"multi_gpu": True, "reduction": "sum"},
            binary_data(y_true4),
            binary_data(y_pred4),
            segments4,
            None,
            5e-6,
        ),
        (
            HingeMarginLoss,
            {"min_margin": 0.7 * 2.0, "reduction": "sum"},
            binary_data(y_true4),
            binary_data(y_pred4),
            segments4,
            None,
            5e-6,
        ),
        (
            HKRLoss,
            {
                "alpha": 2.5,
                "min_margin": 2.0,
                "reduction": "sum",
                "multi_gpu": True,
            },
            binary_data(y_true4),
            binary_data(y_pred4),
            segments4,
            None,
            5e-6,
        ),
        (
            TauBinaryCrossentropyLoss,
            {"tau": 1.5, "reduction": "sum"},
            binary_data(y_true4),
            binary_data(y_pred4),
            segments4,
            None,
            5e-6,
        ),
        (
            KRMulticlassLoss,
            {"multi_gpu": True, "reduction": "sum"},
            y_true3,
            y_pred3,
            [0, 3, 7, None],
            np.float32(698 / 75),
            1e-7,
        ),  # OK (-1,1) but no more one hot
        (
            HingeMulticlassLoss,
            {"min_margin": 2.0, "reduction": "sum"},
            y_true3,
            y_pred3,
            [0, 3, 7, None],
            np.float32(43 / 6),
            1e-7,
        ),  # OK (-1,1) but no more one hot
        (
            MultiMarginLoss,
            {"min_margin": 0.7, "reduction": "sum"},
            y_true3,
            y_pred3,
            [0, 3, 7, None],
            np.float32(9 / 5),
            1e-7,
        ),  # OK (-1,1) but no more one hot
        (
            HKRMulticlassLoss,
            {"multi_gpu": True, "alpha": 2.5, "min_margin": 1.0, "reduction": "sum"},
            y_true3,
            y_pred3,
            [0, 3, 7, None],
            np.float32(152 / 75) * uft.scaleAlpha(2.5),
            1e-7,
        ),  # OK (-1,1) but no more one hot
        (
            KRMulticlassLoss,
            {"multi_gpu": True, "reduction": "sum"},
            binary_data(y_true4),
            binary_data(y_pred4),
            segments4,
            None,
            5e-6,
        ),  # OK (-1,1) but no more one hot
        (
            HingeMulticlassLoss,
            {"min_margin": 2.0, "reduction": "sum"},
            binary_data(y_true4),
            binary_data(y_pred4),
            segments4,
            None,
            5e-6,
        ),  # OK (-1,1) but no more one hot
        (
            MultiMarginLoss,
            {"min_margin": 0.7, "reduction": "sum"},
            binary_data(y_true4),
            binary_data(y_pred4),
            segments4,
            None,
            5e-6,
        ),  # OK (-1,1) but no more one hot
        (
            HKRMulticlassLoss,
            {"multi_gpu": True, "alpha": 2.5, "min_margin": 1.0, "reduction": "sum"},
            binary_data(y_true4),
            binary_data(y_pred4),
            segments4,
            None,
            5e-6,
        ),  # OK (-1,1) but no more one hot
    ],
)
def test_minibatches_binary_loss_generic(
    loss_instance, loss_params, y_true_np, y_pred_np, segments, expected_loss, rtol
):
    # def test_minibatches_binary_losses(self):
    """
    Assert binary losses with mini-batches, to simulate multi-GPU/TPU. Three losses
    are tested using "sum" reduction: KRLoss, HingeMarginLoss and HKRLoss.
    Assertions are tested with:
    - losses on hardcoded small data of size [8, 1] with full batch and mini-batches
    - losses on large random data with full batch and mini-batches.
    """

    if hasattr(loss_instance, "unavailable_class"):
        pytest.skip(f"{loss_instance} not implemented")

    loss = uft.get_instance_framework(loss_instance, inst_params=loss_params)
    if loss is None:
        pytest.skip(f"{loss_instance}  with params {loss_params} not implemented")
    y_true, y_pred = uft.to_tensor(y_true_np), uft.to_tensor(y_pred_np)
    y_true = process_labels_for_multi_gpu(y_true)
    loss_val = uft.compute_loss(loss, y_pred, y_true).numpy()
    if expected_loss is not None:
        np.testing.assert_allclose(
            loss_val,
            expected_loss,
            rtol=rtol,
            atol=5e-6,
            err_msg=f"Loss {loss} failed",
        )
    loss_val_minibatches = 0
    for i in range(len(segments) - 1):
        loss_val_minibatches += uft.compute_loss(
            loss,
            y_pred[segments[i] : segments[i + 1]],
            y_true[segments[i] : segments[i + 1]],
        )

    np.testing.assert_allclose(
        loss_val,
        loss_val_minibatches,
        rtol=rtol,  # 5e-6,
        atol=5e-6,
        err_msg=f"Loss {loss} failed for hardcoded mini-batches",
    )


@pytest.mark.parametrize(
    "loss_instance, loss_params, rtol",
    [
        (
            KRLoss,
            {"name": "KRLoss none", "reduction": "none"},
            1e-7,
        ),
        (
            HingeMarginLoss,
            {"min_margin": 0.4, "reduction": "none", "name": "hinge none"},
            1e-7,
        ),
        (
            HKRLoss,
            {
                "alpha": 5.2,
                "reduction": "none",
                "name": "HKRLoss none",
            },
            1e-7,
        ),
        (
            KRLoss,
            {"name": "KRLoss auto", "reduction": "auto"},
            1e-7,
        ),
        (
            HingeMarginLoss,
            {"min_margin": 0.4, "reduction": "auto", "name": "hinge auto"},
            1e-7,
        ),
        (
            HKRLoss,
            {
                "alpha": 10.0,
                "reduction": "auto",
                "name": "HKRLoss auto",
            },
            1e-7,
        ),
        (
            KRLoss,
            {
                "multi_gpu": True,
                "name": "KRLoss multi_gpu",
                "reduction": "sum",
            },
            1e-7,
        ),
        (
            HKRLoss,
            {
                "alpha": 3.2,
                "multi_gpu": True,
                "reduction": "sum",
                "name": "HKRLoss multi_gpu",
            },
            1e-7,
        ),
    ],
)
def test_multilabel_loss_generic(loss_instance, loss_params, rtol):
    """
    Assert binary losses with multilabels.
    Three losses are tested (KRLoss, HingeMarginLoss and HKRLoss).
    We compare losses with three separate binary classification and
    the corresponding multilabel problem.
    """
    # Create predictions and labels for 3 binary problems and the concatenated
    # multilabel one.

    if hasattr(loss_instance, "unavailable_class"):
        pytest.skip(f"{loss_instance} not implemented")

    y_pred1_np, y_true1_np = get_gaussian_data(1000)
    y_pred2_np, y_true2_np = get_gaussian_data(1000)
    y_pred3_np, y_true3_np = get_gaussian_data(1000)

    y_pred_np = np.concatenate([y_pred1_np, y_pred2_np, y_pred3_np], axis=-1)
    y_true_np = np.concatenate([y_true1_np, y_true2_np, y_true3_np], axis=-1)

    loss = uft.get_instance_framework(loss_instance, inst_params=loss_params)
    if loss is None:
        pytest.skip(f"{loss_instance}  with params {loss_params} not implemented")

    y_true1, y_pred1 = uft.to_tensor(y_true1_np), uft.to_tensor(y_pred1_np)
    loss_val1 = uft.compute_loss(loss, y_pred1, y_true1).numpy()
    y_true2, y_pred2 = uft.to_tensor(y_true2_np), uft.to_tensor(y_pred2_np)
    loss_val2 = uft.compute_loss(loss, y_pred2, y_true2).numpy()
    y_true3, y_pred3 = uft.to_tensor(y_true3_np), uft.to_tensor(y_pred3_np)
    loss_val3 = uft.compute_loss(loss, y_pred3, y_true3).numpy()
    mean_loss_vals = (loss_val1 + loss_val2 + loss_val3) / 3

    y_true, y_pred = uft.to_tensor(y_true_np), uft.to_tensor(y_pred_np)
    loss_val_multilabel = uft.compute_loss(loss, y_pred, y_true).numpy()

    np.testing.assert_allclose(
        loss_val_multilabel,
        mean_loss_vals,
        rtol=rtol,  # 5e-6,
        atol=1e-4,
        err_msg=f"Loss {loss} failed",
    )
