import numpy as np
import tensorflow as tf

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
