import numpy as np


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
