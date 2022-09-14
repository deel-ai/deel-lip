import tensorflow as tf

if tf.__version__.startswith("2.0"):
    from tensorflow.python.framework.random_seed import set_seed
else:
    set_seed = tf.random.set_seed
import numpy as np


def set_global_seed(seed):
    np.random.seed(seed)
    set_seed(seed)
