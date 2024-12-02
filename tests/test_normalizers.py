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
from functools import partial

import numpy as np

from . import utils_framework as uft

from .utils_framework import (
    tLinear,
    tConv2d,
    bjorck_normalization,
    reshaped_kernel_orthogonalization,
    spectral_normalization,
    spectral_normalization_conv,
    DEFAULT_EPS_SPECTRAL,
)

from .utils_framework import (
    bjorck_norm,
    remove_bjorck_norm,
    frobenius_norm,
    remove_frobenius_norm,
    compute_lconv_coef,
    lconv_norm,
    remove_lconv_norm,
)
from .utils_framework import _padding_circular

rng = np.random.default_rng(42)


@pytest.mark.parametrize(
    "kernel_shape",
    [
        (15, 32),
        (32, 15),
    ],
)
def test_kernel_svd(kernel_shape):
    """Compare max singular value using power iteration and np.linalg.svd"""
    kernel = rng.normal(size=kernel_shape).astype("float32")
    sigmas_svd = np.linalg.svd(
        np.reshape(kernel, (np.prod(kernel.shape[:-1]), kernel.shape[-1])),
        full_matrices=False,
        compute_uv=False,
    )
    SVmax = np.max(sigmas_svd)

    u = rng.normal(size=(1, kernel.shape[-1])).astype("float32")

    kernel = uft.to_tensor(kernel)
    u = uft.to_tensor(u)
    W_bar, _u, sigma = uft.get_instance_framework(
        spectral_normalization,
        {"kernel": kernel, "u": u, "eps": DEFAULT_EPS_SPECTRAL},
    )
    # Test sigma is close to the one computed with svd first run @ 1e-1
    np.testing.assert_approx_equal(
        sigma, SVmax, 1, "test failed with kernel_shape " + str(kernel.shape)
    )

    W_bar, _u, sigma = uft.get_instance_framework(
        spectral_normalization,
        {"kernel": kernel, "u": _u, "eps": DEFAULT_EPS_SPECTRAL},
    )
    # spectral_normalization(kernel, u=_u, eps=1e-6)
    W_bar, _u, sigma = uft.get_instance_framework(
        spectral_normalization,
        {"kernel": kernel, "u": _u, "eps": DEFAULT_EPS_SPECTRAL},
    )
    # spectral_normalization(kernel, u=_u, eps=1e-6)
    # Test W_bar is reshaped correctly
    np.testing.assert_equal(W_bar.shape, (np.prod(kernel.shape[:-1]), kernel.shape[-1]))
    # Test sigma is close to the one computed with svd second run @ 1e-5
    np.testing.assert_approx_equal(
        sigma, SVmax, 3, "test failed with kernel_shape " + str(kernel.shape)
    )
    # Test if kernel is normalized by sigma
    np.testing.assert_allclose(
        np.reshape(W_bar, kernel.shape), kernel / sigmas_svd[0], atol=1e-2
    )


def set_spectral_input_shape(kernel, strides):
    """Set spectral input shape and RO_case, depending on kernel shape and
    strides."""
    (kh, kw, c_in, c_out) = kernel.shape
    cPad = [kh // 2, kw // 2]
    stride = strides[0]

    # Compute minimal N
    r = kh // 2
    if r < 1:
        N = 5
    else:
        N = 4 * r + 1
        if stride > 1:
            N = int(0.5 + N / stride)

    if c_in * stride**2 > c_out:
        spectral_input_shape = (N, N, c_out)
        RO_case = True
    else:
        spectral_input_shape = (stride * N, stride * N, c_in)
        RO_case = False
    return spectral_input_shape, RO_case, cPad


@pytest.mark.parametrize(
    "kernel_shape, strides",
    [
        ((5, 5, 32, 64), (1, 1)),
        ((3, 3, 12, 8), (1, 1)),
        ((3, 3, 24, 24), (1, 1)),
    ],
)
def test_kernel_conv_svd(kernel_shape, strides):
    """Compare power iteration conv against SVD."""

    if hasattr(spectral_normalization_conv, "unavailable_class"):
        pytest.skip("spectral_normalization_conv not implemented")
    if hasattr(_padding_circular, "unavailable_class"):
        pytest.skip("_padding_circular not implemented")

    np.random.seed(42)

    kernel = np.random.normal(size=kernel_shape).astype("float32")
    spectral_input_shape, RO_case, cPad = set_spectral_input_shape(kernel, strides)

    # Compute max singular value using FFT2 and SVD
    kernel_n = kernel.astype(dtype="float32")
    transforms = np.fft.fft2(
        kernel_n,
        (spectral_input_shape[0], spectral_input_shape[1]),
        axes=[0, 1],
    )
    svd = np.linalg.svd(transforms, compute_uv=False)
    SVmax = np.max(svd)

    # Compute max singular value using power iteration conv
    _u = np.random.normal(size=(1,) + spectral_input_shape).astype("float32")
    fPad = partial(_padding_circular, circular_paddings=cPad)

    kernel = uft.to_tensor(kernel)
    _u = uft.to_tensor(_u)
    res = uft.get_instance_framework(
        spectral_normalization_conv,
        {
            "kernel": kernel,
            "u": _u,
            "stride": strides[0],
            "conv_first": not RO_case,
            "pad_func": fPad,
            "eps": 1e-6,
            "maxiter": 30,
        },
    )
    if res is None:
        pytest.skip("spectral_normalization_conv with params not available")
    W_bar, _u, sigma = res

    # Test if sigma is close to the one computed with svd first run @ 1e-1
    np.testing.assert_approx_equal(
        sigma, SVmax, 1, "test failed with kernel_shape " + str(kernel.shape)
    )

    # Run a second time power iteration conv with last _u from first run
    W_bar, _u, sigma = uft.get_instance_framework(
        spectral_normalization_conv,
        {
            "kernel": kernel,
            "u": _u,
            "stride": strides[0],
            "conv_first": not RO_case,
            "pad_func": fPad,
            "eps": 1e-6,
            "maxiter": 30,
        },
    )
    # Test if W_bar is reshaped correctly
    np.testing.assert_equal(W_bar.shape, kernel.shape)
    # Test if sigma is close to the one computed with svd, second run
    np.testing.assert_approx_equal(
        sigma, SVmax, 2, "test failed with kernel_shape " + str(kernel.shape)
    )
    # Test if kernel is normalized by sigma
    np.testing.assert_allclose(
        np.reshape(W_bar, kernel.shape), kernel / SVmax, atol=1e-2
    )


@pytest.mark.parametrize(
    "kernel_shape",
    [
        (15, 32),
        (64, 32),
    ],
)
def test_bjorck_normalization(kernel_shape):
    np.random.seed(42)

    kernel = np.random.normal(size=kernel_shape).astype("float32")
    """Compare max singular value using power iteration and tf.linalg.svd"""
    sigmas_svd = np.linalg.svd(
        np.reshape(kernel, (np.prod(kernel.shape[:-1]), kernel.shape[-1])),
        full_matrices=False,
        compute_uv=False,
    )
    SVmax = np.max(sigmas_svd)

    kernel = uft.to_tensor(kernel)
    wbar = uft.get_instance_framework(
        bjorck_normalization, {"w": kernel / SVmax, "eps": 1e-5}
    )
    # wbar = bjorck_normalization(kernel / SVmax, eps=1e-5)
    sigmas_wbar_svd = np.linalg.svd(
        np.reshape(wbar, (np.prod(wbar.shape[:-1]), wbar.shape[-1])),
        full_matrices=False,
        compute_uv=False,
    )
    # Test sigma is close to the one computed with svd first run @ 1e-1
    np.testing.assert_allclose(
        sigmas_wbar_svd, np.ones(sigmas_wbar_svd.shape), atol=1e-2
    )
    # Test W_bar is reshaped correctly
    np.testing.assert_equal(wbar.shape, (np.prod(kernel.shape[:-1]), kernel.shape[-1]))

    # Test sigma is close to the one computed with svd second run @1e-5
    wbar = uft.get_instance_framework(bjorck_normalization, {"w": wbar, "eps": 1e-5})
    # wbar = bjorck_normalization(wbar, eps=1e-5)
    sigmas_wbar_svd = np.linalg.svd(
        np.reshape(wbar, (np.prod(wbar.shape[:-1]), wbar.shape[-1])),
        full_matrices=False,
        compute_uv=False,
    )
    np.testing.assert_allclose(
        sigmas_wbar_svd, np.ones(sigmas_wbar_svd.shape), atol=1e-4
    )


@pytest.mark.parametrize(
    "kernel_shape",
    [
        (15, 32),
        (5, 5, 64, 32),
    ],
)
def test_reshaped_kernel_orthogonalization(kernel_shape):
    if hasattr(reshaped_kernel_orthogonalization, "unavailable_class"):
        pytest.skip("reshaped_kernel_orthogonalization not implemented")
    np.random.seed(42)

    kernel = np.random.normal(size=kernel_shape).astype("float32")
    """Compare max singular value using power iteration and tf.linalg.svd"""
    sigmas_svd = np.linalg.svd(
        np.reshape(kernel, (np.prod(kernel.shape[:-1]), kernel.shape[-1])),
        full_matrices=False,
        compute_uv=False,
    )
    SVmax = np.max(sigmas_svd)

    kernel = uft.to_tensor(kernel)
    res = uft.get_instance_framework(
        reshaped_kernel_orthogonalization,
        {
            "kernel": kernel,
            "u": None,
            "adjustment_coef": 1.0,
            "eps_spectral": 1e-5,
            "eps_bjorck": 1e-5,
        },
    )
    if res is None:
        pytest.skip("reshaped_kernel_orthogonalization with params not available")
    W_bar, _, sigma = res
    # Test W_bar is reshaped correctly
    np.testing.assert_equal(W_bar.shape, kernel.shape)
    # Test RKO sigma is close to max(svd)
    np.testing.assert_approx_equal(
        sigma, SVmax, 1, "test failed with kernel_shape " + str(kernel.shape)
    )
    sigmas_wbar_svd = np.linalg.svd(
        np.reshape(W_bar, (np.prod(W_bar.shape[:-1]), W_bar.shape[-1])),
        full_matrices=False,
        compute_uv=False,
    )
    # Test if SVs of W_bar are close to one
    np.testing.assert_allclose(
        sigmas_wbar_svd, np.ones(sigmas_wbar_svd.shape), atol=1e-2
    )


@pytest.mark.skipif(
    hasattr(bjorck_norm, "unavailable_class"),
    reason="bjorck_norm not available",
)
def test_bjorck_norm():
    """
    test bjorck_norm parametrization implementation
    """
    np.random.seed(42)

    m = uft.get_instance_framework(
        tLinear, {"in_features": 2, "out_features": 2}
    )  # torch.nn.Linear(2, 2)
    # torch.nn.init.orthogonal_(m.weight)
    w1 = uft.get_instance_framework(
        bjorck_normalization, {"w": uft.get_layer_weights(m), "eps": 1e-5}
    )
    # bjorck_normalization(m.weight)

    # bjorck norm parametrization
    uft.get_instance_framework(bjorck_norm, {"module": m})  # (m)

    # ensure that the original weight is the only torch parameter
    uft.check_parametrization(m, is_parametrized=True)
    # assert isinstance(m.parametrizations.weight.original, torch.nn.Parameter)
    # assert not isinstance(m.weight, torch.nn.Parameter)

    # check the orthogonality of the weight
    x = np.random.rand(2)
    x = uft.to_tensor(x)
    m(x)
    np.testing.assert_allclose(
        uft.to_numpy(w1), uft.to_numpy(uft.get_layer_weights(m)), atol=1e-5
    )

    # remove the parametrization
    uft.get_instance_framework(remove_bjorck_norm, {"module": m})  # (m)
    uft.check_parametrization(m, is_parametrized=False)
    # assert not hasattr(m, "parametrizations")
    # assert isinstance(m.weight, torch.nn.Parameter)
    np.testing.assert_allclose(
        uft.to_numpy(w1), uft.to_numpy(uft.get_layer_weights(m)), atol=1e-5
    )


@pytest.mark.skipif(
    hasattr(frobenius_norm, "unavailable_class"),
    reason="frobenius_norm not available",
)
def test_frobenius_norm():
    """
    test frobenius_norm parametrization implementation
    """
    np.random.seed(42)

    m = uft.get_instance_framework(
        tLinear, {"in_features": 2, "out_features": 2}
    )  # torch.nn.Linear(2, 2)
    # torch.nn.init.uniform_(m.weight)
    w1 = uft.to_numpy(uft.get_layer_weights(m))
    w1 = w1 / np.linalg.norm(w1)  # m.weight / torch.norm(m.weight)

    # frobenius norm parametrization
    uft.get_instance_framework(frobenius_norm, {"module": m, "disjoint_neurons": False})

    # ensure that the original weight is the only torch parameter
    uft.check_parametrization(m, is_parametrized=True)
    # assert isinstance(m.parametrizations.weight.original, torch.nn.Parameter)
    # assert not isinstance(m.weight, torch.nn.Parameter)

    # check the orthogonality of the weight
    x = np.random.rand(2)
    x = uft.to_tensor(x)
    m(x)
    np.testing.assert_allclose(
        uft.to_numpy(w1), uft.to_numpy(uft.get_layer_weights(m)), atol=1e-5
    )

    # remove the parametrization
    uft.get_instance_framework(remove_frobenius_norm, {"module": m})
    uft.check_parametrization(m, is_parametrized=False)
    np.testing.assert_allclose(
        uft.to_numpy(w1), uft.to_numpy(uft.get_layer_weights(m)), atol=1e-5
    )


@pytest.mark.skipif(
    hasattr(frobenius_norm, "unavailable_class"),
    reason="frobenius_norm not available",
)
def test_frobenius_norm_disjoint_neurons():
    """
    Test `disjoint_neurons=True` argument in frobenius_norm parametrization
    """
    np.random.seed(42)

    params = {"in_features": 5, "out_features": 3}
    m = uft.get_instance_framework(tLinear, params)

    # Set parametrization and perform a forward pass to compute new weights
    uft.get_instance_framework(frobenius_norm, {"module": m, "disjoint_neurons": True})

    x = np.random.rand(5)
    x = uft.to_tensor(x)
    m(x)

    # Assert that all rows of matrix weight are independently normalized
    ww = uft.to_numpy(uft.get_layer_weights(m))
    for i in range(params["out_features"]):
        np.testing.assert_allclose(np.linalg.norm(ww[i, :]), 1.0, rtol=2e-7)


@pytest.mark.skipif(
    hasattr(lconv_norm, "unavailable_class"),
    reason="lconv_norm not available",
)
def test_lconv_norm():
    """
    test lconv_norm parametrization implementation
    """
    np.random.seed(42)

    params = {
        "in_channels": 1,
        "out_channels": 2,
        "kernel_size": (3, 3),
        "stride": (1, 1),
    }
    m = uft.get_instance_framework(tConv2d, params)
    # torch.nn.init.orthogonal_(m.weight)
    w1 = uft.get_layer_weights(m) * compute_lconv_coef(
        params["kernel_size"], None, params["stride"]
    )

    # lconv norm parametrization
    uft.get_instance_framework(lconv_norm, {"module": m})

    shape = [1, 1, 5, 5]
    shape = uft.to_framework_channel(shape)
    x = np.random.rand(*shape)
    x = uft.to_tensor(x)
    _ = m(x)

    # ensure that the original weight is the only torch parameter
    uft.check_parametrization(m, is_parametrized=True)
    # assert isinstance(m.parametrizations.weight.original, torch.nn.Parameter)
    # assert not isinstance(m.weight, torch.nn.Parameter)

    # check the normalization of the weight
    np.testing.assert_allclose(
        uft.to_numpy(w1), uft.to_numpy(uft.get_layer_weights(m)), atol=1e-7
    )
    # tt.assert_equal(y, torch.nn.functional.conv2d(x, w1, bias=m.bias, stride=(1, 1)))

    # remove the parametrization
    uft.get_instance_framework(remove_lconv_norm, {"module": m})
    uft.check_parametrization(m, is_parametrized=False)
    # assert isinstance(m.weight, torch.nn.Parameter)
    np.testing.assert_allclose(
        uft.to_numpy(w1), uft.to_numpy(uft.get_layer_weights(m)), atol=1e-7
    )
