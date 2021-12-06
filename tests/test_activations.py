from unittest import TestCase
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from deel.lip.activations import GroupSort
import os
import numpy as np


def check_serialization(layer):
    m = Sequential([Input(10), layer])
    m.compile(optimizer=SGD(), loss=CategoricalCrossentropy(from_logits=False))
    name = layer.__class__.__name__
    path = os.path.join("logs", "losses", name)
    x = tf.random.uniform((255, 10), -10, 10)
    y1 = m(x)
    m.save(path)
    m2 = load_model(path, compile=True)
    y2 = m2(x)
    np.testing.assert_allclose(y1.numpy(), y2.numpy())


class TestGroupSort(TestCase):
    def test_simple(self):
        gs = GroupSort(n=2)
        check_serialization(gs)
        gs = GroupSort(n=5)
        check_serialization(gs)

    def test_flat_input(self):
        gs = GroupSort(n=2)
        x = tf.convert_to_tensor(
            [
                [1, 2, 3, 4],
                [4, 3, 2, 1],
                [1, 2, 1, 2],
                [2, 1, 2, 1],
            ],
            dtype=tf.float32,
        )
        gs.build(tf.TensorShape((None, 4)))
        y = gs(x).numpy()
        y_t = [
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 4.0, 1.0, 2.0],
            [1.0, 2.0, 1.0, 2.0],
            [1.0, 2.0, 1.0, 2.0],
        ]
        np.testing.assert_equal(y, y_t)

    def test_gs4(self):
        gs = GroupSort(n=4)
        x = tf.convert_to_tensor(
            [
                [1, 2, 3, 4],
                [4, 3, 2, 1],
                [1, 2, 1, 2],
                [2, 1, 2, 1],
            ],
            dtype=tf.float32,
        )
        gs.build(tf.TensorShape((None, 4)))
        y = gs(x).numpy()
        y_t = [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
        ]
        np.testing.assert_equal(y, y_t)

    def test_img_input(self):
        gs = GroupSort(n=2)
        x = tf.convert_to_tensor(
            [
                [1, 2, 3, 4],
                [4, 3, 2, 1],
                [1, 2, 1, 2],
                [2, 1, 2, 1],
            ],
            dtype=tf.float32,
        )
        x = tf.repeat(tf.expand_dims(tf.repeat(tf.expand_dims(x, 1), 28, 1), 1), 28, 1)
        gs.build(tf.TensorShape((None, 28, 28, 4)))
        y = gs(x).numpy()
        y_t = [
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 4.0, 1.0, 2.0],
            [1.0, 2.0, 1.0, 2.0],
            [1.0, 2.0, 1.0, 2.0],
        ]
        y_t = tf.repeat(
            tf.expand_dims(tf.repeat(tf.expand_dims(y_t, 1), 28, 1), 1), 28, 1
        )
        np.testing.assert_equal(y, y_t)

    def test_idempotence(self):
        gs = GroupSort(n=2)
        x = tf.random.uniform((255, 16), -10, 10)
        gs.build(tf.TensorShape((None, 16)))
        y1 = gs(x)
        y2 = gs(y1)
        np.testing.assert_equal(y1.numpy(), y2.numpy())
        gs = GroupSort(n=2)
        x = tf.random.uniform((255, 16), -10, 10)
        gs.build(tf.TensorShape((None, 16)))
        y1 = gs(x)
        y2 = gs(y1)
        np.testing.assert_equal(y1.numpy(), y2.numpy())
