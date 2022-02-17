import numpy as np
import tensorflow as tf

from .activations import GroupSort, MaxMin
from .layers import (
    Condensable,
    SpectralConv2D,
    FrobeniusConv2D,
    LorthRegulConv2D,
    SpectralDense,
    FrobeniusDense,
)
from .normalizers import _power_iteration_conv


# Not the best place and no the best code => may replace by wandb
def printAndLog(txt, log_out=None):
    print(txt)
    if log_out is not None:
        print(txt, file=log_out)


def zero_upscale2D(x, strides):
    stride_v = strides[0] * strides[1]
    if stride_v == 1:
        return x
    output_shape = x.get_shape().as_list()[1:]
    if strides[1] > 1:
        output_shape[1] *= strides[1]
        x = tf.expand_dims(x, 3)
        fillz = tf.zeros_like(x)
        fillz = tf.tile(fillz, [1, 1, 1, strides[1] - 1, 1])
        x = tf.concat((x, fillz), axis=3)
        x = tf.reshape(x, (-1,) + tuple(output_shape))
    if strides[0] > 1:
        output_shape[0] *= strides[0]
        x = tf.expand_dims(x, 2)
        fillz = tf.zeros_like(x)
        fillz = tf.tile(fillz, [1, 1, strides[0] - 1, 1, 1])
        x = tf.concat((x, fillz), axis=2)
        x = tf.reshape(x, (-1,) + tuple(output_shape))
    return x


def transposeKernel(w, transpose=False):
    if not transpose:
        return w
    wAdj = tf.transpose(w, perm=[0, 1, 3, 2])
    wAdj = wAdj[::-1, ::-1, :]
    return wAdj


def compute_layer_vs_2D(w, Ks, N, nbIter):
    (R0, R, d, D) = w.shape
    KN = int(Ks * N)
    batch_size = 1
    cPad = [int(R0 / 2), int(R / 2)]

    if Ks * Ks * d > D:
        input_shape = (N, N, D)
        conv_first = False
    else:
        input_shape = (KN, KN, d)
        conv_first = True

    # Maximum singular value

    u = tf.random.uniform((batch_size,) + input_shape, minval=-1.0, maxval=1.0)

    u, v = _power_iteration_conv(
        w, u, stride=Ks, conv_first=conv_first, cPad=cPad, niter=nbIter
    )

    sigma_max = tf.norm(v)  # norm_u(v)
    # print(tf.norm(v))

    # Minimum Singular Value

    bigConstant = 1.1 * sigma_max**2
    print("bigConstant " + str(bigConstant))
    u = tf.random.uniform((batch_size,) + input_shape, minval=-1.0, maxval=1.0)

    u, v = _power_iteration_conv(
        w,
        u,
        stride=Ks,
        conv_first=conv_first,
        cPad=cPad,
        bigConstant=bigConstant,
        niter=nbIter,
    )

    if bigConstant - tf.norm(u) >= 0:  # cas normal
        sigma_min = tf.sqrt(bigConstant - tf.norm(u))
    elif (
        bigConstant - tf.norm(u) >= -0.0000000000001
    ):  # précaution pour gérer les erreurs numériques
        sigma_min = 0
    else:
        sigma_min = -1  # veut dire qu'il y a un prolème

    return (float(sigma_min), float(sigma_max))


def computeDenseSV(layer, input_sizes, numIter=100, log_out=None):
    weights = np.copy(layer.get_weights()[0])
    printAndLog("----------------------------------------------------------", log_out)
    printAndLog(
        "Layer type " + str(type(layer)) + " weight shape " + str(weights.shape),
        log_out,
    )
    new_w = weights  # np.reshape(weights, [weights.shape[0], -1])
    svdtmp = np.linalg.svd(new_w, compute_uv=False)
    SVmin = np.min(svdtmp)
    SVmax = np.max(svdtmp)
    printAndLog(
        "kernel(W) SV (min, max, mean) " + str((SVmin, SVmax, np.mean(svdtmp))), log_out
    )
    return (SVmin, SVmax)


def computeConvSV(layer, input_sizes, numIter=100, log_out=None):
    Ks = layer.strides[0]
    assert isinstance(input_sizes, tuple)
    input_size = input_sizes[1]
    # isLinear = False
    weights = np.copy(layer.get_weights()[0])
    # print(weights.shape)
    kernel_n = weights.astype(dtype="float32")
    if Ks > 1:
        # print("Warning np.linalg.svd incompatible with strides")
        SVmin, SVmax = compute_layer_vs_2D(weights, Ks, input_size, numIter)
        printAndLog(
            "Conv(K) SV min et max [conv iter]: " + str((SVmin, SVmax)), log_out
        )
        # out_stats['conv SV (min, max, mean)']=(vs[0],vs[1],-1234)
    else:
        kernel_n = weights.astype(dtype="float32")
        transforms = np.fft.fft2(kernel_n, (input_size, input_size), axes=[0, 1])
        svd = np.linalg.svd(transforms, compute_uv=False)
        SVmin = np.min(svd)
        SVmax = np.max(svd)
        printAndLog(
            "Conv(K) SV min et max [np.linalg.svd]: " + str((SVmin, SVmax)), log_out
        )
        printAndLog(
            "Conv(K) SV mean et std [np.linalg.svd]: "
            + str((np.mean(svd), np.std(svd))),
            log_out,
        )
        # print("SV ",np.sort(np.reshape(svd,(-1,))))
    return (SVmin, SVmax)


# Warning this is not SV for non linear functions but grad min and grad max
def computeActivationSV(layer, input_sizes=[], numIter=100, log_out=None):
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


def addSV(layer, input_sizes=[], numIter=100, log_out=None):
    assert isinstance(input_sizes, list)
    return (1.0, 1.0) * len(input_sizes)


def bnSV(layer, input_sizes=[], numIter=100, log_out=None):
    values = np.abs(
        layer.gamma.numpy() / np.sqrt(layer.moving_variance.numpy() + layer.epsilon)
    )
    upper = np.max(values)
    lower = np.min(values)
    return (lower, upper)


def compute_layer_sv(
    layer, input_sizes=[], numIter=100, log_out=None, supplementaryType2SV={}
):
    defaultType2SV = {
        tf.keras.layers.Conv2D: computeConvSV,
        tf.keras.layers.Conv2DTranspose: computeConvSV,
        SpectralConv2D: computeConvSV,
        FrobeniusConv2D: computeConvSV,
        LorthRegulConv2D: computeConvSV,
        tf.keras.layers.Dense: computeDenseSV,
        SpectralDense: computeDenseSV,
        FrobeniusDense: computeDenseSV,
        tf.keras.layers.ReLU: computeActivationSV,
        tf.keras.layers.Activation: computeActivationSV,
        GroupSort: computeActivationSV,
        MaxMin: computeActivationSV,
        tf.keras.layers.Add: addSV,
        tf.keras.layers.BatchNormalization: bnSV,
    }
    src_layer = layer
    if isinstance(layer, Condensable):
        printAndLog("vanilla_export", log_out)
        printAndLog(str(type(layer)), log_out)
        layer.condense()
        layer = layer.vanilla_export()
    printAndLog("----------------------------------------------------------", log_out)
    if type(layer) in defaultType2SV.keys():
        lower_upper = defaultType2SV[type(layer)](
            layer, src_layer.input_shape, numIter=numIter, log_out=log_out
        )
    elif type(layer) in supplementaryType2SV.keys():
        lower_upper = supplementaryType2SV[type(layer)](
            layer, src_layer.input_shape, numIter=numIter, log_out=log_out
        )
    else:
        printAndLog("No SV for layer " + str(type(layer)), log_out)
        lower_upper = (None, None)
    printAndLog(
        "Layer type " + str(type(layer)) + " (upper,lower) = " + str(lower_upper),
        log_out,
    )
    return lower_upper


def computeModelSVs(
    model, input_sizes=[], numIter=100, log_out=None, supplementaryType2SV={}
):
    list_SV = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.models.Model) or isinstance(
            layer, tf.keras.models.Sequential
        ):
            list_SV.append((layer.name, (None, None)))
            list_SV += computeModelSVs(
                layer,
                input_sizes=input_sizes,
                numIter=numIter,
                log_out=log_out,
                supplementaryType2SV=supplementaryType2SV,
            )
        else:
            list_SV.append(
                (
                    layer.name,
                    compute_layer_sv(
                        layer,
                        input_sizes=input_sizes,
                        numIter=numIter,
                        log_out=log_out,
                        supplementaryType2SV=supplementaryType2SV,
                    ),
                )
            )
    return list_SV


def generate_graph_layers(model, layerName, nodeN=0, outputN=-1, strName=None):
    def addLayersOutput(lay, nodeN, outputN=-1):
        dict_output2layName = {}
        outs = lay.get_output_at(nodeN)
        if isinstance(outs, list):
            lay_output = outs
        else:
            lay_output = [outs]
        if outputN < 0:
            for ll in lay_output:
                dict_output2layName[ll.name] = lay.name
        else:
            dict_output2layName[lay_output[outputN].name] = lay.name
        return dict_output2layName

    layers = model.layers
    firstLay = model.get_layer(layerName)
    listLayersOutputs = addLayersOutput(firstLay, nodeN, outputN)
    listLayers = [firstLay.name]
    listNodes = [nodeN]
    dictInputLayers = {}
    print("Start layer " + firstLay.name)
    print("Start layer (Node " + str(nodeN) + ") output" + str(firstLay.output))
    for lay in layers:
        listInputLayers = []
        for nn in range(len(lay.inbound_nodes)):
            ins = lay.get_input_at(nn)
            # print("layer intput"+str(lay.input))
            if isinstance(ins, list):
                lay_input = ins
            else:
                lay_input = [ins]
            # print(lay_input)
            for ii in lay_input:
                if ii.name in listLayersOutputs.keys():
                    # print("new layer "+lay.name+" node "+str(nn))
                    listLayers.append(lay.name)
                    listLayersOutputs.update(addLayersOutput(lay, nn))
                    listNodes.append(nn)
                    listInputLayers.append(listLayersOutputs[ii.name])
                    # print(listLayersOutputs)
        dictInputLayers[lay.name] = listInputLayers
    print(dictInputLayers)
    return dictInputLayers


def computeModelUpperLip(
    model, input_size=-1, numIter=100, log_out=None, supplementaryType2SV={}
):
    list_SV = computeModelSVs(
        model, input_sizes=[], numIter=100, log_out=None, supplementaryType2SV={}
    )
    dictInputLayers = generate_graph_layers(model, layerName=model.layers[1].name)
    UpperLip = 1.0
    LowerLip = 1.0
    count_nb_notknown = 0
    dict_cumulatedSV = {}
    for svs in list_SV:
        inpLayers = dictInputLayers[svs[0]]
        if len(inpLayers) == 0:
            UpperLip = 1.0
            LowerLip = 1.0
        else:
            UpperLip = 0.0
            LowerLip = 0.0
            if svs[1][0] is not None:
                for ii, iLay in enumerate(inpLayers):
                    UpperLip += dict_cumulatedSV[iLay][1] * svs[1][2 * ii + 1]
                    LowerLip += dict_cumulatedSV[iLay][0] * svs[1][2 * ii + 0]
            else:
                for ii, iLay in enumerate(inpLayers):
                    UpperLip += dict_cumulatedSV[iLay][1]
                    LowerLip += dict_cumulatedSV[iLay][0]
                count_nb_notknown += 1
        dict_cumulatedSV[svs[0]] = (LowerLip, UpperLip)
    last_layer = list_SV[-1][0]
    printAndLog(
        "Cumulated lower and upper gradient bound "
        + str(last_layer)
        + ": "
        + str(LowerLip)
        + ", "
        + str(UpperLip),
        log_out,
    )
    return list_SV, dict_cumulatedSV
