import numpy as np
import tensorflow as tf

from .layers import GroupSort, MaxMin


def _compute_sv_dense(layer, input_sizes=None):
    """Compute max and min singular values for a Dense layer.

    The singular values are computed using the SVD decomposition of the weight matrix.

    Args:
        layer (tf.keras.Layer): the Dense layer.
        input_sizes (tuple, optional): unused here.

    Returns:
        tuple: min and max singular values
    """
    weights = layer.get_weights()[0]
    svd = np.linalg.svd(weights, compute_uv=False)
    return (np.min(svd), np.max(svd))


def _generate_conv_matrix(layer, input_sizes):
    """Generate equivalent matrix for a convolutional layer.

    The convolutional layer is converted to a dense layer by computing the equivalent
    matrix. The equivalent matrix is computed by applying the convolutional layer on a
    dirac input.

    Args:
        layer (tf.keras.Layer): the convolutional layer to convert to dense.
        input_sizes (tuple): the input shape of the layer (with batch dimension as first
            element).

    Returns:
        np.array: the equivalent matrix of the convolutional layer.
    """
    single_layer_model = tf.keras.models.Sequential(
        [tf.keras.layers.Input(input_sizes[1:]), layer]
    )
    dirac_inp = np.zeros((input_sizes[2],) + input_sizes[1:])  # Line by line generation
    in_size = input_sizes[1] * input_sizes[2]
    channel_in = input_sizes[-1]
    w_eqmatrix = None
    start_index = 0
    for ch in range(channel_in):
        for ii in range(input_sizes[1]):
            dirac_inp[:, ii, :, ch] = np.eye(input_sizes[2])
            out_pred = single_layer_model(dirac_inp)
            if w_eqmatrix is None:
                w_eqmatrix = np.zeros(
                    (in_size * channel_in, np.prod(out_pred.shape[1:]))
                )
            w_eqmatrix[start_index : (start_index + input_sizes[2]), :] = tf.reshape(
                out_pred, (input_sizes[2], -1)
            )
            dirac_inp = 0.0 * dirac_inp
            start_index += input_sizes[2]
    return w_eqmatrix


def _compute_sv_conv2d_layer(layer, input_sizes):
    """Compute max and min singular values for any convolutional layer.

    The convolutional layer is converted to a dense layer by computing the equivalent
    matrix. The equivalent matrix is computed by applying the convolutional layer on a
    dirac input. The singular values are then computed using the SVD decomposition of
    the weight matrix.

    Args:
        layer (tf.keras.Layer): the convolutional layer.
        input_sizes (tuple): the input shape of the layer (with batch dimension as first
            element).

    Returns:
        tuple: min and max singular values
    """
    w_eqmatrix = _generate_conv_matrix(layer, input_sizes)
    svd = np.linalg.svd(w_eqmatrix, compute_uv=False)
    return (np.min(svd), np.max(svd))


def _compute_sv_activation(layer, input_sizes=None):
    """Compute min and max gradient norm for activation.

    Warning: This is not singular values for non-linear functions but gradient norm.
    """
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


def _compute_sv_add(layer, input_sizes):
    """Compute min and max singular values of Add layer."""
    assert isinstance(input_sizes, list)
    return (len(input_sizes) * 1.0, len(input_sizes) * 1.0)


def _compute_sv_bn(layer, input_sizes=None):
    """Compute min and max singular values of BatchNormalization layer."""
    values = np.abs(
        layer.gamma.numpy() / np.sqrt(layer.moving_variance.numpy() + layer.epsilon)
    )
    return (np.min(values), np.max(values))
