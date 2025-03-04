import copy
import os
import warnings
from functools import partial
import numpy as np

import keras.utils as K
import tensorflow as tf

from keras.models import Model as tModel

from keras.models import Sequential as tSequential

from keras.saving import load_model as tload_model

from keras.layers import Input as tInput

from keras.optimizers import SGD, Adam

from keras.losses import CategoricalCrossentropy as CategoricalCrossentropy

from keras.layers import Layer as tLayer

from keras.layers import Dense as tLinear

from keras.layers import Flatten

from keras.utils import set_random_seed as set_seed

from keras.metrics import MeanSquaredError as tmse
from keras.losses import MeanSquaredError as MeanSquaredError

from tensorflow import int32 as type_int32

from keras.losses import Loss

from keras.layers import Add as tAdd

from keras.layers import ReLU as tReLU

from keras.layers import Activation as tActivation

from keras.layers import Softmax as tSoftmax

from keras.layers import Reshape as tReshape

from keras.layers import MaxPool2D as tMaxPool2d

from keras.layers import Conv2D as tConv2d

from keras.layers import UpSampling2D as tUpSampling2d

from keras.layers import Concatenate as tConcatenate

from deel.lip.activations import GroupSort as GroupSort
from deel.lip.activations import GroupSort2 as GroupSort2
from deel.lip.activations import Householder as HouseHolder
from deel.lip.layers import LipschitzLayer

from deel.lip.constraints import (
    AutoWeightClipConstraint,
    SpectralConstraint,
    FrobeniusConstraint,
)

from deel.lip.callbacks import CondenseCallback, MonitorCallback
from deel.lip.layers import (
    InvertibleDownSampling,
    InvertibleUpSampling,
)
from deel.lip.layers import SpectralDense as SpectralLinear
from deel.lip.layers import FrobeniusDense as FrobeniusLinear
from deel.lip.layers import SpectralConv2D as SpectralConv2d
from deel.lip.layers import FrobeniusConv2D as FrobeniusConv2d
from deel.lip.layers import SpectralConv2DTranspose as SpectralConvTranspose2d
from deel.lip.layers import ScaledAveragePooling2D as ScaledAvgPool2d
from deel.lip.layers import ScaledGlobalAveragePooling2D as ScaledAdaptiveAvgPool2d
from deel.lip.layers import ScaledL2NormPooling2D as ScaledL2NormPool2d
from deel.lip.layers import ScaledGlobalL2NormPooling2D as ScaledAdaptativeL2NormPool2d
from deel.lip.model import Sequential, Model
from deel.lip.utils import evaluate_lip_const, process_labels_for_multi_gpu
from deel.lip import vanillaModel

from deel.lip.losses import KR as KRLoss
from deel.lip.losses import HingeMargin as HingeMarginLoss
from deel.lip.losses import HKR as HKRLoss
from deel.lip.losses import MulticlassKR as KRMulticlassLoss
from deel.lip.losses import MulticlassHinge as HingeMulticlassLoss
from deel.lip.losses import MulticlassHKR as HKRMulticlassLoss

from deel.lip.losses import MultiMargin as MultiMarginLoss
from deel.lip.losses import TauCategoricalCrossentropy as TauCategoricalCrossentropyLoss
from deel.lip.losses import (
    TauSparseCategoricalCrossentropy as TauSparseCategoricalCrossentropyLoss,
)
from deel.lip.losses import TauBinaryCrossentropy as TauBinaryCrossentropyLoss
from deel.lip.losses import CategoricalHinge as CategoricalHingeLoss
from deel.lip.initializers import SpectralInitializer


from deel.lip.metrics import (
    CategoricalProvableRobustAccuracy,
    BinaryProvableRobustAccuracy,
    CategoricalProvableAvgRobustness,
    BinaryProvableAvgRobustness,
)


from deel.lip.normalizers import (
    bjorck_normalization,
    reshaped_kernel_orthogonalization,
    spectral_normalization,
    spectral_normalization_conv,
)
from deel.lip.normalizers import DEFAULT_MAXITER_SPECTRAL as DEFAULT_NITER_SPECTRAL_INIT
from deel.lip.normalizers import DEFAULT_EPS_SPECTRAL, DEFAULT_EPS_BJORCK
from deel.lip.utils import _padding_circular

from deel.lip.regularizers import Lorth2D as Lorth2d
from deel.lip.regularizers import LorthRegularizer
from deel.lip.layers.unconstrained import PadConv2D as PadConv2d

from deel.lip.compute_layer_sv import compute_layer_sv
from deel.lip.regularizers import OrthDenseRegularizer as OrthLinearRegularizer

framework = "keras3"

# to avoid linter F401
__all__ = [
    "tModel",
    "Flatten",
    "tAdd",
    "tReLU",
    "tActivation",
    "tSoftmax",
    "tReshape",
    "tUpSampling2d",
    "tConcatenate",
    "type_int32",
    "GroupSort2",
    "HouseHolder",
    "AutoWeightClipConstraint",
    "SpectralConstraint",
    "FrobeniusConstraint",
    "CondenseCallback",
    "MonitorCallback",
    "Sequential",
    "ScaledAdaptativeL2NormPool2d",
    "evaluate_lip_const",
    "DEFAULT_NITER_SPECTRAL_INIT",
    "Loss",
    "process_labels_for_multi_gpu",
    "vanillaModel",
    "CategoricalCrossentropy",
    "MeanSquaredError",
    "HingeMarginLoss",
    "KRMulticlassLoss",
    "HingeMulticlassLoss",
    "HKRMulticlassLoss",
    "SoftHKRMulticlassLoss",
    "MultiMarginLoss",
    "TauCategoricalCrossentropyLoss",
    "TauSparseCategoricalCrossentropyLoss",
    "TauBinaryCrossentropyLoss",
    "CategoricalHingeLoss",
    "SpectralInitializer",
    "CategoricalProvableRobustAccuracy",
    "BinaryProvableRobustAccuracy",
    "CategoricalProvableAvgRobustness",
    "BinaryProvableAvgRobustness",
    "bjorck_normalization",
    "reshaped_kernel_orthogonalization",
    "spectral_normalization_conv",
    "Lorth2d",
    "LorthRegularizer",
    "compute_layer_sv",
    "OrthLinearRegularizer",
    "DEFAULT_EPS_SPECTRAL",
    "DEFAULT_EPS_BJORCK",
    "set_seed",
]

FIT = "fit"
EVALUATE = "evaluate"

MODEL_PATH = "model"
EXTENSION = ".keras"
LIP_LAYERS = "lip_layers"


# not implemented
class module_Unavailable_class:
    def __init__(self, **kwargs):
        self.unavailable = True
        return None

    def unavailable_class():
        return True

    def __call__(self, **kwargs):
        return None


invertible_downsample = module_Unavailable_class
invertible_upsample = module_Unavailable_class
bjorck_norm = remove_bjorck_norm = frobenius_norm = remove_frobenius_norm = (
    compute_lconv_coef
) = lconv_norm = remove_lconv_norm = module_Unavailable_class

SpectralConv1d = module_Unavailable_class
LipResidual = module_Unavailable_class
BatchCentering = module_Unavailable_class
LayerCentering = module_Unavailable_class
tSplit = module_Unavailable_class
SoftHKRMulticlassLoss = module_Unavailable_class


def get_instance_generic(instance_type, inst_params):
    return instance_type(**inst_params)


def get_optim_generic(instance_type, inst_params):
    layp = copy.deepcopy(inst_params)
    layp.pop("model", None)
    if "lr" in layp:
        layp["learning_rate"] = layp.pop("lr")
    return instance_type(**layp)


def replace_key_params(inst_params, dict_keys_replace):
    layp = copy.deepcopy(inst_params)
    for k, v in dict_keys_replace.items():
        if k in layp:
            val = layp.pop(k)
            if v is None:
                warnings.warn(
                    UserWarning("Warning key is not used", k, " in tensorflow")
                )
            else:
                if isinstance(v, tuple):
                    layp[v[0]] = v[1](val)
                else:
                    layp[v] = val

    return layp


def get_instance_withreplacement(instance_type, inst_params, dict_keys_replace):
    layp = replace_key_params(inst_params, dict_keys_replace)
    return instance_type(**layp)


def get_instance_withcheck(
    instance_type, inst_params, dict_keys_replace={}, list_keys_notimplemented=[]
):
    for k in list_keys_notimplemented:
        if isinstance(k, tuple):
            kk = k[0]
            kv = k[1]
        else:
            kk = k
            kv = None
        if kk in inst_params:
            if (kv is None) or inst_params[kk] in kv:
                warnings.warn(
                    UserWarning("Warning key is not implemented", kk, " in tensorflow")
                )
                return None
    layp = replace_key_params(inst_params, dict_keys_replace)
    return instance_type(**layp)


getters_dict = {
    GroupSort: partial(
        get_instance_withreplacement, dict_keys_replace={"group_size": "n"}
    ),
    SpectralConvTranspose2d: partial(
        get_instance_withreplacement,
        dict_keys_replace={
            "in_channels": None,
            "out_channels": "filters",
            "bias": "use_bias",
            "padding": ("padding", lambda x: "valid" if x == 0 else "same"),
            "padding_mode": None,
            "dilation": "dilation_rate",
            "stride": "strides",
        },
    ),
    ScaledAdaptiveAvgPool2d: partial(
        get_instance_withreplacement, dict_keys_replace={"output_size": None}
    ),
    ScaledAvgPool2d: partial(
        get_instance_withreplacement,
        dict_keys_replace={
            "kernel_size": "pool_size",
            "stride": "strides",
        },
    ),
    ScaledL2NormPool2d: partial(
        get_instance_withreplacement,
        dict_keys_replace={"kernel_size": "pool_size", "stride": "strides"},
    ),
    ScaledAdaptativeL2NormPool2d: partial(
        get_instance_withreplacement, dict_keys_replace={"output_size": None}
    ),
    SpectralConv2d: partial(
        get_instance_withcheck,
        dict_keys_replace={
            "in_channels": None,
            "out_channels": "filters",
            "padding": None,
            "padding_mode": (
                "padding",
                lambda x: "same" if x == "zeros" else "not implemented",
            ),
            "bias": "use_bias",
            "stride": "strides",
        },
        list_keys_notimplemented=[
            ("padding_mode", ["reflect", "symmetric", "circular"])
        ],
    ),
    SpectralLinear: partial(
        get_instance_withreplacement,
        dict_keys_replace={
            "in_features": None,
            "out_features": "units",
            "bias": "use_bias",
        },
    ),
    FrobeniusLinear: partial(
        get_instance_withreplacement,
        dict_keys_replace={
            "in_features": None,
            "out_features": "units",
            "bias": "use_bias",
        },
    ),
    FrobeniusConv2d: partial(
        get_instance_withcheck,
        dict_keys_replace={
            "in_channels": None,
            "out_channels": "filters",
            "padding": None,
            "padding_mode": (
                "padding",
                lambda x: "same" if x == "zeros" else "not implemented",
            ),
            "bias": "use_bias",
            "stride": "strides",
        },
        list_keys_notimplemented=[
            ("padding_mode", ["reflect", "symmetric", "circular"])
        ],
    ),
    InvertibleDownSampling: partial(
        get_instance_withreplacement, dict_keys_replace={"kernel_size": "pool_size"}
    ),
    InvertibleUpSampling: partial(
        get_instance_withreplacement, dict_keys_replace={"kernel_size": "pool_size"}
    ),
    tMaxPool2d: partial(
        get_instance_withreplacement,
        dict_keys_replace={"kernel_size": "pool_size", "stride": "strides"},
    ),
    tLinear: partial(
        get_instance_withreplacement,
        dict_keys_replace={
            "in_features": None,
            "out_features": "units",
            "bias": "use_bias",
        },
    ),
    SGD: get_optim_generic,
    Adam: get_optim_generic,
    KRLoss: partial(
        get_instance_withreplacement,
        dict_keys_replace={
            "true_values": None,
            "reduction": (
                "reduction",
                lambda x: "sum_over_batch_size" if x.lower() in ["auto"] else x,
            ),
        },
    ),
    HKRLoss: partial(
        get_instance_withreplacement,
        dict_keys_replace={
            "true_values": None,
            "reduction": (
                "reduction",
                lambda x: "sum_over_batch_size" if x.lower() in ["auto"] else x,
            ),
        },
    ),
    HingeMarginLoss: partial(
        get_instance_withreplacement,
        dict_keys_replace={
            "true_values": None,
            "reduction": (
                "reduction",
                lambda x: "sum_over_batch_size" if x.lower() in ["auto"] else x,
            ),
        },
    ),
    spectral_normalization: partial(
        get_instance_withreplacement, dict_keys_replace={"niter": "maxiter"}
    ),
    tConv2d: partial(
        get_instance_withreplacement,
        dict_keys_replace={
            "in_channels": None,
            "out_channels": "filters",
            "padding": ("padding", lambda x: "valid" if x == 0 else "same"),
            "bias": "use_bias",
            "padding_mode": None,
            "stride": "strides",
        },
    ),
    PadConv2d: partial(
        get_instance_withreplacement,
        dict_keys_replace={
            "in_channels": None,
            "out_channels": "filters",
            "padding": None,
            "bias": "use_bias",
            "padding_mode": "padding",
        },
    ),
    HouseHolder: partial(
        get_instance_withreplacement, dict_keys_replace={"channels": None}
    ),
    tReshape: partial(
        get_instance_withreplacement,
        dict_keys_replace={"dim": None, "unflattened_size": "target_shape"},
    ),
    KRMulticlassLoss: partial(
        get_instance_withreplacement,
        dict_keys_replace={
            "reduction": (
                "reduction",
                lambda x: "sum_over_batch_size" if x.lower() in ["auto"] else x,
            ),
        },
    ),
}


def get_instance_framework(instance_type, inst_params):
    if instance_type not in getters_dict:
        instance = get_instance_generic(instance_type, inst_params)
    else:
        instance = getters_dict[instance_type](instance_type, inst_params)
    if instance is None:
        print("instance is not implemented", instance_type)
    return instance


def generate_k_lip_model(layer_type: type, layer_params: dict, input_shape=None, k=1):
    """
    build a model with a single layer of given type, with defined lipshitz factor.

    Args:
        layer_type: the type of layer to use
        layer_params: parameter passed to constructor of layer_type
        input_shape: the shape of the input
        k: lipshitz factor of the function

    Returns:
        a keras Model with a single layer.

    """
    if issubclass(layer_type, tSequential):
        model = layer_type(**layer_params)
        if k is not None:
            model.set_klip_factor(k)
        return model
    a = tInput(shape=input_shape)
    if issubclass(layer_type, LipschitzLayer):
        layer_params["k_coef_lip"] = k
    layer = get_instance_framework(layer_type, layer_params)
    assert isinstance(layer, tLayer)
    b = layer(a)
    return Model(inputs=a, outputs=b)


def get_functional_model(modeltype, dict_tensors, functional_input_output_tensors):
    inputs, outputs = functional_input_output_tensors(dict_tensors, None)
    return modeltype(inputs=inputs, outputs=outputs)


def build_named_sequential(modeltype, dict_layers):
    list_lay = []
    for n, v in dict_layers.items():
        v.name = n
        list_lay.append(v)
    return modeltype(list_lay)


def init_session():
    K.clear_session()
    return


def compute_output_shape(input_shape, model):
    output_shape = model.compute_output_shape((1,) + input_shape)[1:]
    return output_shape


# compute_loss
def compute_loss(loss, y_pred, y_true):
    return loss(y_true, y_pred)


def compile_model(model, loss, optimizer, metrics=[]):
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return loss, optimizer, metrics


def build_layer(layer, input_shape):
    layer.build(tf.TensorShape((None,) + input_shape))


def to_tensor(nparray, dtype=tf.float32):
    return tf.convert_to_tensor(nparray, dtype=dtype)


def to_numpy(tens):
    if isinstance(tens, tf.Tensor):
        return tens.numpy()
    else:
        return tens


def save_model(model, path, overwrite=True):
    if not path.endswith(EXTENSION):
        path = path + EXTENSION
    if overwrite:
        if os.path.exists(path):
            os.remove(path)
    parent_dirpath = os.path.split(path)[0]
    if not os.path.exists(parent_dirpath):
        os.makedirs(parent_dirpath)

    model.save(path)
    return


def load_model(
    path, compile=True, layer_type=None, layer_params=True, input_shape=None, k=None
):
    if not path.endswith(EXTENSION):
        path = path + EXTENSION
    return tload_model(path, compile=compile)


def get_layer_weights_by_index(model, layer_idx):
    return get_layer_weights(model.layers[layer_idx + 1])  # Input layer in Tf


# .weight.detach().cpu().numpy()


def get_layer_weights(layer):
    return layer.kernel


def get_children(model):
    return model.layers


def get_named_children(model):
    return [(layer.name, layer) for layer in model.layers]


def initialize_kernel(model, layer_idx, kernel_initializer):
    return  # Done in layer co,nstructor


def initializers_Constant(value):
    return tf.keras.initializers.Constant(value)


def compute_predict(model, x, training=False):
    return model(x, training=training)


def train(
    train_dl,
    model,
    loss_fn,
    optimizer,
    epoch,
    batch_size,
    steps_per_epoch,
    callbacks=[],
):
    model.__getattribute__(FIT)(
        train_dl,
        steps_per_epoch=steps_per_epoch,
        epochs=epoch,
        verbose=1,
        callbacks=callbacks,
    )


def run_test(model, test_dl, loss_fn, metrics, steps=10):
    loss, mse = model.__getattribute__(EVALUATE)(
        test_dl,
        steps=steps,
    )
    return loss, mse


def to_framework_channel(x):  # channel first to channel last
    if len(x) != 3:
        return x
    else:
        return (x[1], x[2], x[0])


def to_NCHW(x):
    return np.transpose(x, (0, 3, 1, 2))


def to_NCHW_inv(x):
    return np.transpose(x, (0, 2, 3, 1))


def get_NCHW(x):
    return (x.shape[0], x.shape[3], x.shape[1], x.shape[2])


def metric_mse():
    return tmse()


def scaleAlpha(alpha):
    warnings.warn("scaleAlpha is deprecated, use alpha in [0,1] instead")
    return 1.0
    # return (1.0/(1+1.0/alpha))


def scaleDivAlpha(alpha):
    warnings.warn("scaleDivAlpha is deprecated, use alpha in [0,1] instead")
    return 1.0


def vanilla_require_a_copy():
    return False


def copy_model_parameters(model_src, model_dest):
    model_dest.set_weights(model_src.get_weights())


def is_supported_padding(padding, layer_type):
    layertype2padding = {
        SpectralConv2d: ["same", "zeros"],
        FrobeniusConv2d: ["same", "zeros"],
        PadConv2d: ["same", "valid", "constant", "reflect", "symmetric", "circular"],
    }
    if layer_type in layertype2padding:
        return padding.lower() in layertype2padding[layer_type]
    else:
        assert False
        warnings.warn(f"layer {layer_type} type not supported for padding")
        return False


def pad_input(x, padding, kernel_size):
    """Pad an input tensor x with corresponding padding, based on kernel size."""
    if isinstance(kernel_size, (int, float)):
        kernel_size = [kernel_size, kernel_size]
    if padding.lower() in ["same", "valid"]:
        return x
    elif padding.lower() in ["constant", "reflect", "symmetric"]:
        p_vert, p_hor = kernel_size[0] // 2, kernel_size[1] // 2
        pad_sizes = [[0, 0], [p_vert, p_vert], [p_hor, p_hor], [0, 0]]
        return tf.pad(x, pad_sizes, padding)
    elif padding.lower() in ["circular"]:
        p_vert, p_hor = kernel_size[0] // 2, kernel_size[1] // 2
        return _padding_circular(x, (p_vert, p_hor))


def check_parametrization(m, is_parametrized):
    assert True  # No parametrization in Tensorflow
