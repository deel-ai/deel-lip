from unittest import TestCase
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD
from deel.lip.utils import load_model
import os
import numpy as np

# a tester:
# - un cas hardcodé
# - les dtypes pour y_true
# - shape y_pred
# - labels en [0,1] et en [-1,1]
# - liens entre les losses
from deel.lip.losses import *


def get_gaussian_data(n=500, mean1=1.0, mean2=-1.0):
    x1 = tf.random.normal((n, 1), mean=1.0, stddev=1.0)
    x2 = tf.random.normal((n, 1), mean=-1.0, stddev=1.0)
    y_pred = tf.concat([x1, x2], axis=0)
    y_true = tf.concat([tf.ones(n), tf.zeros(n)], axis=0)
    return y_pred, y_true


def check_serialization(nb_class, loss):
    m = Sequential([Input(10), Dense(nb_class)])
    m.compile(optimizer=SGD(), loss=loss)
    path = os.path.join("logs", "losses", loss.__name__)
    m.save(path)
    m2 = load_model(path, compile=True)
    m2(tf.random.uniform((255, 10)))


class Test(TestCase):
    def test_kr_loss(self):
        loss = KR_loss
        y_pred, y_true = get_gaussian_data(5000, 1.0, -1.0)
        l = loss(y_true, y_pred).numpy()
        np.testing.assert_almost_equal(
            l, 2.0, 1, "test failed when y_true has shape (bs, )"
        )
        l2 = loss(
            tf.expand_dims(y_true, axis=-1), tf.expand_dims(y_pred, axis=-1)
        ).numpy()
        np.testing.assert_almost_equal(
            l2, 2.0, 1, "test failed when y_true has shape (bs, 1)"
        )
        l3 = loss(tf.cast(y_true, dtype=tf.int32), y_pred).numpy()
        np.testing.assert_almost_equal(
            l3, 2.0, 1, "test failed when y_true has dtype int32"
        )
        check_serialization(1, loss)

    def test_neg_kr_loss(self):
        loss = neg_KR_loss
        y_pred, y_true = get_gaussian_data(5000, 1.0, -1.0)
        l = loss(y_true, y_pred).numpy()
        np.testing.assert_almost_equal(
            l, -2.0, 1, "test failed when y_true has shape (bs, )"
        )
        l2 = loss(
            tf.expand_dims(y_true, axis=-1), tf.expand_dims(y_pred, axis=-1)
        ).numpy()
        np.testing.assert_almost_equal(
            l2, -2.0, 1, "test failed when y_true has shape (bs, 1)"
        )
        l3 = loss(tf.cast(y_true, dtype=tf.int32), y_pred).numpy()
        np.testing.assert_almost_equal(
            l3, -2.0, 1, "test failed when y_true has dtype int32"
        )
        check_serialization(1, loss)

    def test_hkr_loss(self):
        self.fail()

    def test_hinge_margin_loss(self):
        self.fail()

    def test_kr_multiclass_loss(self):
        self.fail()

    def test_hinge_multiclass_loss(self):
        self.fail()

    def test_hkr_multiclass_loss(self):
        self.fail()

    def test_multi_margin_loss(self):
        self.fail()
