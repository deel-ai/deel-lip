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

from . import utils_framework as uft

from .utils_framework import (
    SpectralInitializer,
    tLinear,
)


@pytest.mark.parametrize(
    "layer_type, layer_params,input_shape, orthogonal_test",
    [
        (
            tLinear,
            {
                "in_features": 5,
                "out_features": 4,
                "kernel_initializer": uft.get_instance_framework(
                    SpectralInitializer,
                    inst_params={"eps_spectral": 1e-6, "eps_bjorck": 1e-6},
                ),
            },
            (5,),
            True,
        ),
        (
            tLinear,
            {
                "in_features": 5,
                "out_features": 100,
                "kernel_initializer": uft.get_instance_framework(
                    SpectralInitializer,
                    inst_params={"eps_spectral": 1e-6, "eps_bjorck": None},
                ),
            },
            (5,),
            False,
        ),
    ],
)
def test_initializer(layer_type, layer_params, input_shape, orthogonal_test):
    np.random.seed(42)
    # clear session to avoid side effects from previous train
    uft.init_session()  # K.clear_session()
    input_shape = uft.to_framework_channel(input_shape)
    # create the model, defin opt, and compile it
    model = uft.generate_k_lip_model(layer_type, layer_params, input_shape)
    uft.initialize_kernel(model, 0, layer_params["kernel_initializer"])

    optimizer = uft.get_instance_framework(
        uft.Adam, inst_params={"lr": 0.001, "model": model}
    )  # uft.Adam(lr=0.001)

    loss_fn, optimizer, metrics = uft.compile_model(
        model,
        optimizer=optimizer,
        loss=uft.MeanSquaredError(),
        metrics=[uft.metric_mse()],
    )

    sigmas = np.linalg.svd(
        uft.to_numpy(uft.get_layer_weights_by_index(model, 0)),
        full_matrices=False,
        compute_uv=False,
    )
    if orthogonal_test:
        np.testing.assert_allclose(sigmas, np.ones_like(sigmas), atol=1e-5)
    else:
        np.testing.assert_allclose(sigmas.max(), 1.0, atol=2e-2)
