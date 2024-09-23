import unittest

import keras
import numpy as np

from deel.lip.layers.unconstrained import PadConv2D
from deel.lip.model import vanillaModel
from deel.lip.utils import _padding_circular

from .test_layers import generate_k_lip_model


class TestPadConv2D(unittest.TestCase):
    def test_PadConv2D(self):
        """Main test for PadConv2D: tests on padding, predict and vanilla export."""

        all_paddings = ("circular", "constant", "symmetric", "reflect", "same", "valid")
        layer_params = {
            "filters": 2,
            "kernel_size": (3, 3),
            "use_bias": False,
            "padding": "valid",
        }
        batch_size = 250
        kernel_input_shapes = [
            [(3, 3), 2, (5, 5, 1)],
            [(3, 3), 2, (5, 5, 5)],
            [(7, 7), 2, (7, 7, 2)],
            [3, 2, (5, 5, 128)],
        ]

        for k_shape, filters, input_shape in kernel_input_shapes:
            layer_params["kernel_size"] = k_shape
            layer_params["filters"] = filters
            for pad in ["circular", "constant", "symmetric", "reflect"]:
                self._test_padding(pad, input_shape, batch_size, k_shape)
            for pad in all_paddings:
                self._test_predict(layer_params, pad, input_shape, batch_size)
            for pad in all_paddings:
                self._test_vanilla(layer_params, pad, input_shape, batch_size)

    def pad_input(self, x, padding, kernel_size):
        """Pad an input tensor x with corresponding padding, based on kernel size."""
        if isinstance(kernel_size, (int, float)):
            kernel_size = [kernel_size, kernel_size]
        if padding.lower() in ["same", "valid"]:
            return x
        elif padding.lower() in ["constant", "reflect", "symmetric"]:
            p_vert, p_hor = kernel_size[0] // 2, kernel_size[1] // 2
            pad_sizes = [[0, 0], [p_vert, p_vert], [p_hor, p_hor], [0, 0]]
            return keras.ops.pad(x, pad_sizes, padding)
        elif padding.lower() in ["circular"]:
            p_vert, p_hor = kernel_size[0] // 2, kernel_size[1] // 2
            return _padding_circular(x, (p_vert, p_hor))

    def compare(self, x, x_ref, index_x=[], index_x_ref=[]):
        """Compare a tensor and its padded version, based on index_x and ref."""
        x_cropped = x[:, index_x[0] : index_x[1], index_x[3] : index_x[4], :][
            :, :: index_x[2], :: index_x[5], :
        ]
        if index_x_ref[0] is None:  # compare with 0
            np.testing.assert_allclose(x_cropped, np.zeros(x_cropped.shape), 1e-2, 0)
        else:
            np.testing.assert_allclose(
                x_cropped,
                x_ref[
                    :,
                    index_x_ref[0] : index_x_ref[1],
                    index_x_ref[3] : index_x_ref[4],
                    :,
                ][:, :: index_x_ref[2], :: index_x_ref[5], :],
                1e-2,
                0,
            )

    def _test_padding(self, padding_tested, input_shape, batch_size, kernel_size):
        """Test different padding types: assert values in original and padded tensors"""
        kernel_size_list = kernel_size
        if isinstance(kernel_size, (int, float)):
            kernel_size_list = [kernel_size, kernel_size]

        x = np.random.normal(size=(batch_size,) + input_shape).astype("float32")
        x_pad = self.pad_input(x, padding_tested, kernel_size)
        p_vert, p_hor = kernel_size_list[0] // 2, kernel_size_list[1] // 2

        center_x_pad = [p_vert, -p_vert, 1, p_hor, -p_hor, 1, "center"]
        upper_x_pad = [0, p_vert, 1, p_hor, -p_hor, 1, "upper"]
        lower_x_pad = [-p_vert, x_pad.shape[1], 1, p_hor, -p_hor, 1, "lower"]
        left_x_pad = [p_vert, -p_vert, 1, 0, p_hor, 1, "left"]
        right_x_pad = [p_vert, -p_vert, 1, -p_hor, x_pad.shape[2], 1, "right"]
        all_x = [0, x.shape[1], 1, 0, x.shape[2], 1]
        upper_x = [0, p_vert, 1, 0, x.shape[2], 1]
        upper_x_rev = [0, p_vert, -1, 0, x.shape[2], 1]
        upper_x_refl = [1, p_vert + 1, -1, 0, x.shape[2], 1]
        lower_x = [-p_vert, x.shape[1], 1, 0, x.shape[2], 1]
        lower_x_rev = [-p_vert, x.shape[1], -1, 0, x.shape[2], 1]
        lower_x_refl = [-p_vert - 1, x.shape[1] - 1, -1, 0, x.shape[2], 1]
        left_x = [0, x.shape[1], 1, 0, p_hor, 1]
        left_x_rev = [0, x.shape[1], 1, 0, p_hor, -1]
        left_x_refl = [0, x.shape[1], 1, 1, p_hor + 1, -1]
        right_x = [0, x.shape[1], 1, -p_hor, x.shape[2], 1]
        right_x_rev = [0, x.shape[1], 1, -p_hor, x.shape[2], -1]
        right_x_refl = [0, x.shape[1], 1, -p_hor - 1, x.shape[2] - 1, -1]
        zero_pad = [None, None, None, None]
        pad_tests = [
            {
                "circular": [center_x_pad, all_x],
                "constant": [center_x_pad, all_x],
                "symmetric": [center_x_pad, all_x],
                "reflect": [center_x_pad, all_x],
            },
            {
                "circular": [upper_x_pad, lower_x],
                "constant": [upper_x_pad, zero_pad],
                "symmetric": [upper_x_pad, upper_x_rev],
                "reflect": [upper_x_pad, upper_x_refl],
            },
            {
                "circular": [lower_x_pad, upper_x],
                "constant": [lower_x_pad, zero_pad],
                "symmetric": [lower_x_pad, lower_x_rev],
                "reflect": [lower_x_pad, lower_x_refl],
            },
            {
                "circular": [left_x_pad, right_x],
                "constant": [left_x_pad, zero_pad],
                "symmetric": [left_x_pad, left_x_rev],
                "reflect": [left_x_pad, left_x_refl],
            },
            {
                "circular": [right_x_pad, left_x],
                "constant": [right_x_pad, zero_pad],
                "symmetric": [right_x_pad, right_x_rev],
                "reflect": [right_x_pad, right_x_refl],
            },
        ]

        for test_pad in pad_tests:
            self.compare(
                x_pad,
                x,
                index_x=test_pad[padding_tested][0],
                index_x_ref=test_pad[padding_tested][1],
            )

    def _test_predict(self, layer_params, padding_tested, input_shape, batch_size):
        """Compare predictions between pad+Conv2D and PadConv2D layers."""
        x = np.random.normal(size=(batch_size,) + input_shape).astype("float32")
        x_pad = self.pad_input(x, padding_tested, layer_params["kernel_size"])
        layer_params_ref = layer_params.copy()
        if padding_tested.lower() == "same":
            layer_params_ref["padding"] = "same"

        model_ref = generate_k_lip_model(
            layer_type=keras.layers.Conv2D,
            layer_params=layer_params_ref,
            input_shape=x_pad.shape[1:],
            k=1.0,
        )
        y_ref = model_ref.predict(x_pad)

        layer_params_pad = layer_params.copy()
        layer_params_pad["padding"] = padding_tested
        model = generate_k_lip_model(
            layer_type=PadConv2D,
            layer_params=layer_params_pad,
            input_shape=input_shape,
            k=1.0,
        )

        model.layers[1].kernel.assign(model_ref.layers[1].kernel)
        if model.layers[1].use_bias:
            model.layers[1].bias.assign(model_ref.layers[1].bias)
        y = model.predict(x)

        np.testing.assert_allclose(y_ref, y, 1e-2, 0)

    def _test_vanilla(self, layer_params, padding_tested, input_shape, batch_size):
        """Compare predictions between PadConv2D and its vanilla export."""
        x = np.random.normal(size=(batch_size,) + input_shape).astype("float32")
        layer_params_pad = layer_params.copy()
        layer_params_pad["padding"] = padding_tested
        model = generate_k_lip_model(
            layer_type=PadConv2D,
            layer_params=layer_params_pad,
            input_shape=input_shape,
            k=1.0,
        )
        y = model.predict(x)

        model_v = vanillaModel(model)
        y_v = model_v.predict(x)
        np.testing.assert_allclose(y_v, y, 1e-2, 0)
