from functools import partial
import numpy as np
import tensorflow as tf

from .activations import GroupSort, MaxMin
from .layers import (
    Condensable,
    SpectralConv2D,
    FrobeniusConv2D,
    OrthoConv2D,
    PadConv2D,
    SpectralDense,
    FrobeniusDense,
)
from .normalizers import _power_iteration_conv
from .utils import _padding_circular


# Not the best place and no the best code => may replace by wandb
def _print_and_log(txt, log_out=None, verbose=False):
    if verbose:
        print(txt)
    if log_out is not None:
        print(txt, file=log_out)


def _compute_sv_conv2d(w, Ks, N, padding="circular"):
    (R0, R, d, D) = w.shape
    KN = int(Ks * N)
    batch_size = 1
    cPad = None
    if padding in ["circular"]:
        cPad = [int(R0 / 2), int(R / 2)]

    if Ks * Ks * d > D:
        input_shape = (N, N, D)
        conv_first = False
    else:
        input_shape = (KN, KN, d)
        conv_first = True

    # Maximum singular value

    u = tf.random.uniform((batch_size,) + input_shape, minval=-1.0, maxval=1.0)

    if cPad is not None:
        fPad = partial(_padding_circular, circular_paddings=cPad)
    else:
        fPad = None

    u, v, _ = _power_iteration_conv(
        w, u, stride=Ks, conv_first=conv_first, pad_func=fPad
    )

    sigma_max = tf.norm(v)  # norm_u(v)

    # Minimum Singular Value

    bigConstant = 1.1 * sigma_max**2
    u = tf.random.uniform((batch_size,) + input_shape, minval=-1.0, maxval=1.0)
    u, v, norm_u = _power_iteration_conv(
        w,
        u,
        stride=Ks,
        conv_first=conv_first,
        pad_func=fPad,
        bigConstant=bigConstant,
    )

    if bigConstant - norm_u >= 0:  # normal case
        sigma_min = tf.sqrt(bigConstant - norm_u)
    elif (
        bigConstant - norm_u >= -0.0000000000001
    ):  # margin to take into consideration numrica errors
        sigma_min = 0
    else:
        sigma_min = -1  # assertion (should not occur)

    return (float(sigma_min), float(sigma_max))


def _compute_sv_dense(layer, input_sizes, log_out=None, verbose=False):
    weights = np.copy(layer.get_weights()[0])
    _print_and_log(
        "----------------------------------------------------------",
        log_out,
        verbose=verbose,
    )
    _print_and_log(
        "Layer type " + str(type(layer)) + " weight shape " + str(weights.shape),
        log_out,
        verbose=verbose,
    )
    new_w = weights  # np.reshape(weights, [weights.shape[0], -1])
    svdtmp = np.linalg.svd(new_w, compute_uv=False)
    SVmin = np.min(svdtmp)
    SVmax = np.max(svdtmp)
    _print_and_log(
        "kernel(W) SV (min, max, mean) " + str((SVmin, SVmax, np.mean(svdtmp))),
        log_out,
        verbose=verbose,
    )
    return (SVmin, SVmax)


def _compute_sv_padconv2d(
    layer, input_sizes, padding="circular", log_out=None, verbose=False
):
    Ks = layer.strides[0]
    assert isinstance(input_sizes, tuple)
    input_size = input_sizes[1]
    # isLinear = False
    weights = np.copy(layer.get_weights()[0])
    kernel_n = weights.astype(dtype="float32")
    if (Ks > 1) or (padding not in ["circular"]):
        SVmin, SVmax = _compute_sv_conv2d(weights, Ks, input_size, padding=padding)
        _print_and_log(
            "Conv(K) SV min et max [conv iter]: " + str((SVmin, SVmax)),
            log_out,
            verbose=verbose,
        )
    else:
        # only for circular padding and without stride
        kernel_n = weights.astype(dtype="float32")
        transforms = np.fft.fft2(kernel_n, (input_size, input_size), axes=[0, 1])
        svd = np.linalg.svd(transforms, compute_uv=False)
        SVmin = np.min(svd)
        SVmax = np.max(svd)
        _print_and_log(
            "Conv(K) SV min et max [np.linalg.svd]: " + str((SVmin, SVmax)),
            log_out,
            verbose=verbose,
        )
        _print_and_log(
            "Conv(K) SV mean et std [np.linalg.svd]: "
            + str((np.mean(svd), np.std(svd))),
            log_out,
            verbose=verbose,
        )
    return (SVmin, SVmax)


def _compute_sv_padsame_conv2d(layer, input_sizes, log_out=None):
    return _compute_sv_padconv2d(layer, input_sizes, padding="same", log_out=None)


# Warning this is not SV for non linear functions but grad min and grad max
def _compute_sv_activation(layer, input_sizes=[], log_out=None):
    if isinstance(layer, tf.keras.layers.Activation):
        function2SV = {tf.keras.activations.relu: (0, 1)}
        if layer.activation in function2SV.keys():
            return function2SV[layer.activation]
        else:
            return (None, None)
    layer2SV = {
        tf.keras.layers.ReLU: (0, 1),
        GroupSort: (1, 1),
        MaxMin: (1, 1),
    }
    if layer in layer2SV.keys():
        return layer2SV[layer.activation]
    else:
        return (None, None)


def _compute_sv_add(layer, input_sizes=[], log_out=None):
    assert isinstance(input_sizes, list)
    return (len(input_sizes) * 1.0, len(input_sizes) * 1.0)


def _compute_sv_bn(layer, input_sizes=[], log_out=None):
    values = np.abs(
        layer.gamma.numpy() / np.sqrt(layer.moving_variance.numpy() + layer.epsilon)
    )
    upper = np.max(values)
    lower = np.min(values)
    return (lower, upper)


def compute_layer_sv(layer, supplementary_type2sv={}, log_out=None, verbose=False):
    """
     Compute the largest and lowest singular values (or upper and lower bounds)
     of a given layer
     If case of Condensable layers applies a vanilla_export to the layer
     to get the weights.
     Support by default several kind of layers (Conv2D,Dense,Add, BatchNormalization,
     ReLU, Activation, and deel-lip layers)
    Args:
        layer: a single tf.keras.layer
        supplementary_type2sv: a dictionary linking new layer type with user defined
         function to compute the singular values [optional]
         log_out: file descriptor for dumping verbose information
         verbose: flag to prompt information
    Returns:
        lower_upper (tuple): a tuple (lowest sv, highest sv)
    """
    default_type2sv = {
        tf.keras.layers.Conv2D: _compute_sv_padsame_conv2d,
        tf.keras.layers.Conv2DTranspose: _compute_sv_padsame_conv2d,
        SpectralConv2D: _compute_sv_padsame_conv2d,
        FrobeniusConv2D: _compute_sv_padsame_conv2d,
        PadConv2D: _compute_sv_padconv2d,
        OrthoConv2D: _compute_sv_padconv2d,
        tf.keras.layers.Dense: _compute_sv_dense,
        SpectralDense: _compute_sv_dense,
        FrobeniusDense: _compute_sv_dense,
        tf.keras.layers.ReLU: _compute_sv_activation,
        tf.keras.layers.Activation: _compute_sv_activation,
        GroupSort: _compute_sv_activation,
        MaxMin: _compute_sv_activation,
        tf.keras.layers.Add: _compute_sv_add,
        tf.keras.layers.BatchNormalization: _compute_sv_bn,
    }
    src_layer = layer
    if isinstance(layer, Condensable):
        _print_and_log("vanilla_export", log_out, verbose=verbose)
        _print_and_log(str(type(layer)), log_out, verbose=verbose)
        layer.condense()
        layer = layer.vanilla_export()
    _print_and_log(
        "----------------------------------------------------------",
        log_out,
        verbose=verbose,
    )
    if type(layer) in default_type2sv.keys():
        lower_upper = default_type2sv[type(layer)](
            layer, src_layer.input_shape, log_out=log_out
        )
    elif type(layer) in supplementary_type2sv.keys():
        lower_upper = supplementary_type2sv[type(layer)](
            layer, src_layer.input_shape, log_out=log_out
        )
    else:
        _print_and_log("No SV for layer " + str(type(layer)), log_out, verbose=verbose)
        lower_upper = (None, None)
    _print_and_log(
        "Layer type " + str(type(layer)) + " (upper,lower) = " + str(lower_upper),
        log_out,
        verbose=verbose,
    )
    return lower_upper


def compute_model_sv(model, supplementary_type2sv={}, log_out=None, verbose=False):
    """
     Compute the largest and lowest singular values of all layers in a model
     Args:
        model: a  tf.keras Model or Sequential
        supplementary_type2sv (dict): a dictionary linking new layer type
         with user defined function to compute the singular values [optional]
        log_out: file descriptor for dumping verbose information
        verbose (bool): flag to prompt information
    Returns:
        list_sv (dict): A dictionary indicating for each layer name
        a tuple (lowest sv, highest sv)
    """
    list_sv = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.models.Model) or isinstance(
            layer, tf.keras.models.Sequential
        ):
            list_sv.append((layer.name, (None, None)))
            list_sv += compute_model_sv(
                layer,
                supplementary_type2sv=supplementary_type2sv,
                log_out=log_out,
                verbose=verbose,
            )
        else:
            list_sv.append(
                (
                    layer.name,
                    compute_layer_sv(
                        layer,
                        supplementary_type2sv=supplementary_type2sv,
                        log_out=log_out,
                        verbose=verbose,
                    ),
                )
            )
    return list_sv


def _generate_graph_layers(model, layer_name, node_n=0, output_n=-1):
    """
     Compute the graph of the model
     Args:
        model: a  tf.keras Model or Sequential
        layer_name (str): the name of the first layer
        node_n: node number in layer
        output_n: selection of output branch (-1 for processing any outputs)
    Returns:
        dict_input_layers (dict): A dictionary indicating for each layer name
        a list of its input layers
    """

    def add_layers_output(lay, node_n, output_n=-1):
        dict_output2layname = {}
        outs = lay.get_output_at(node_n)
        if isinstance(outs, list):
            lay_output = outs
        else:
            lay_output = [outs]
        if output_n < 0:
            for ll in lay_output:
                dict_output2layname[ll.name] = lay.name
        else:
            dict_output2layname[lay_output[output_n].name] = lay.name
        return dict_output2layname

    layers = model.layers
    first_lay = model.get_layer(layer_name)
    list_layers_outputs = add_layers_output(first_lay, node_n, output_n)
    list_layers = [first_lay.name]
    list_nodes = [node_n]
    dict_input_layers = {}
    print("Start layer " + first_lay.name)
    print("Start layer (Node " + str(node_n) + ") output" + str(first_lay.output))
    for lay in layers:
        list_input_layers = []
        for nn in range(len(lay.inbound_nodes)):
            ins = lay.get_input_at(nn)
            # print("layer intput"+str(lay.input))
            if isinstance(ins, list):
                lay_input = ins
            else:
                lay_input = [ins]
            # print(lay_input)
            for ii in lay_input:
                if ii.name in list_layers_outputs.keys():
                    # print("new layer "+lay.name+" node "+str(nn))
                    list_layers.append(lay.name)
                    list_layers_outputs.update(add_layers_output(lay, nn))
                    list_nodes.append(nn)
                    list_input_layers.append(list_layers_outputs[ii.name])
                    # print(listLayersOutputs)
        dict_input_layers[lay.name] = list_input_layers
    print(dict_input_layers)
    return dict_input_layers


def compute_model_upper_lip(
    model, supplementary_type2sv={}, log_out=None, verbose=False
):
    """
    Compute the largest and lowest singular values of all layers in a model,
    and cumulated lower and upper values
    Args:
        model: a  tf.keras Model or Sequential
        supplementary_type2sv (dict): a dictionary linking new layer type
         with user defined function to compute the singular values [optional]
        log_out: file descriptor for dumping verbose information
        verbose (bool): flag to prompt information
    Returns:
        list_sv (dict): A dictionary indicating for each layer name
        a tuple (lowest sv, highest sv)
        cumulated_sv (dict): A dictionary indicating for each layer name
        the cumulated, according to teh model graph, lower and upper values

    """
    list_sv = compute_model_sv(
        model, input_sizes=[], log_out=None, supplementary_type2sv=supplementary_type2sv
    )
    index_firstlayer = 0
    if isinstance(model.layers[0], tf.keras.layers.Input):
        index_firstlayer = 1
    dict_input_layers = _generate_graph_layers(
        model, layer_name=model.layers[index_firstlayer].name
    )
    upper_lip = 1.0
    lower_lip = 1.0
    count_nb_notknown = 0
    cumulated_sv = {}
    for svs in list_sv:
        inpLayers = dict_input_layers[svs[0]]
        if len(inpLayers) == 0:
            upper_lip = 1.0
            lower_lip = 1.0
        else:
            upper_lip = 0.0
            lower_lip = 0.0
            if svs[1][0] is not None:
                for ii, iLay in enumerate(inpLayers):
                    upper_lip += cumulated_sv[iLay][1] * svs[1][2 * ii + 1]
                    lower_lip += cumulated_sv[iLay][0] * svs[1][2 * ii + 0]
            else:
                for ii, iLay in enumerate(inpLayers):
                    upper_lip += cumulated_sv[iLay][1]
                    lower_lip += cumulated_sv[iLay][0]
                count_nb_notknown += 1
        cumulated_sv[svs[0]] = (lower_lip, upper_lip)
    last_layer = list_sv[-1][0]
    _print_and_log(
        "Cumulated lower and upper gradient bound "
        + str(last_layer)
        + ": "
        + str(lower_lip)
        + ", "
        + str(upper_lip),
        log_out,
        verbose=verbose,
    )
    return list_sv, cumulated_sv
