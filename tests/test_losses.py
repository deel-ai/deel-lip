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
    x1 = tf.random.normal((n, 1), mean=mean1, stddev=0.1)
    x2 = tf.random.normal((n, 1), mean=mean2, stddev=0.1)
    y_pred = tf.concat([x1, x2], axis=0)
    y_true = tf.concat([tf.ones(n), tf.zeros(n)], axis=0)
    return y_pred, y_true


def check_serialization(nb_class, loss):
    m = Sequential([Input(10), Dense(nb_class)])
    m.compile(optimizer=SGD(), loss=loss)
    name = loss.__class__.__name__ if isinstance(loss, Loss) else loss.__name__
    path = os.path.join("logs", "losses", name)
    m.save(path)
    m2 = load_model(path, compile=True)
    m2(tf.random.uniform((255, 10)))


class Test(TestCase):
    def test_kr_loss(self):
        loss = KR_loss
        y_pred, y_true = get_gaussian_data(5000)
        l = loss(y_true, y_pred).numpy()
        np.testing.assert_approx_equal(
            l, 2.0, 1, "test failed when y_true has shape (bs, )"
        )
        l2 = loss(
            tf.expand_dims(y_true, axis=-1), tf.expand_dims(y_pred, axis=-1)
        ).numpy()
        np.testing.assert_approx_equal(
            l2, 2.0, 1, "test failed when y_true has shape (bs, 1)"
        )
        l3 = loss(tf.cast(y_true, dtype=tf.int32), y_pred).numpy()
        np.testing.assert_approx_equal(
            l3, 2.0, 1, "test failed when y_true has dtype int32"
        )
        y_true2 = tf.where(y_true == 1.0, 1.0, -1.0)
        l4 = loss(y_true2, y_pred).numpy()
        np.testing.assert_approx_equal(
            l4, 2.0, 1, "test failed when labels are in (1, -1)"
        )
        check_serialization(1, loss)

    def test_neg_kr_loss(self):
        loss = neg_KR_loss
        y_pred, y_true = get_gaussian_data(5000)
        l = loss(y_true, y_pred).numpy()
        np.testing.assert_approx_equal(
            l, -2.0, 1, "test failed when y_true has shape (bs, )"
        )
        l2 = loss(
            tf.expand_dims(y_true, axis=-1), tf.expand_dims(y_pred, axis=-1)
        ).numpy()
        np.testing.assert_approx_equal(
            l2, -2.0, 1, "test failed when y_true has shape (bs, 1)"
        )
        l3 = loss(tf.cast(y_true, dtype=tf.int32), y_pred).numpy()
        np.testing.assert_approx_equal(
            l3, -2.0, 1, "test failed when y_true has dtype int32"
        )
        y_true2 = tf.where(y_true == 1.0, 1.0, -1.0)
        l4 = loss(y_true2, y_pred).numpy()
        np.testing.assert_approx_equal(
            l4, -2.0, 1, "test failed when labels are in (1, -1)"
        )
        check_serialization(1, loss)

    def test_hkr_loss(self):
        # assuming KR and hinge have been properly tested
        loss = HKR_loss(0.5, 2.0)
        check_serialization(1, loss)

    def test_hinge_margin_loss(self):
        loss = HingeMarginLoss(1.0)
        y_true = tf.convert_to_tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        y_pred = tf.convert_to_tensor([0.5, 1.5, -0.5, -0.5, -1.5, 0.5])
        l = loss(y_true, y_pred).numpy()
        np.testing.assert_almost_equal(
            l, 0.66, 1, "test failed when y_true has shape (bs, )"
        )
        l2 = loss(
            tf.expand_dims(y_true, axis=-1), tf.expand_dims(y_pred, axis=-1)
        ).numpy()
        np.testing.assert_almost_equal(
            l2, 0.66, 1, "test failed when y_true has shape (bs, 1)"
        )
        l3 = loss(tf.cast(y_true, dtype=tf.int32), y_pred).numpy()
        np.testing.assert_almost_equal(
            l3, 0.66, 1, "test failed when y_true has dtype int32"
        )
        y_true2 = tf.where(y_true == 1.0, 1.0, -1.0)
        l4 = loss(y_true2, y_pred).numpy()
        np.testing.assert_almost_equal(
            l4, 0.66, 1, "test failed when labels are in (1, -1)"
        )
        check_serialization(1, loss)

    def test_kr_multiclass_loss(self):
        multiclass_kr = KRMulticlassLoss(1e-7)
        kr = KR_loss
        y_true = tf.convert_to_tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        y_pred = tf.convert_to_tensor([0.5, 1.5, -0.5, -0.5, -1.5, 0.5])
        l_single = kr(y_true, y_pred).numpy()
        l_multi = multiclass_kr(y_true, y_pred).numpy()
        self.assertEqual(
            l_single,
            l_multi,
            "KR multiclass must yield the same "
            "results when given a single class "
            "vector",
        )
        n_class = 10
        n_items = 100
        y_true = tf.one_hot(np.random.randint(0, 10, n_items), n_class)
        y_pred = tf.random.normal((n_items, n_class))
        l = multiclass_kr(y_true, y_pred).numpy()
        l2 = multiclass_kr(tf.cast(y_true, dtype=tf.int32), y_pred).numpy()
        np.testing.assert_almost_equal(
            l2, l, 1, "test failed when y_true has dtype int32"
        )
        y_true2 = tf.where(y_true == 1.0, 1.0, -1.0)
        l3 = multiclass_kr(y_true2, y_pred).numpy()
        np.testing.assert_almost_equal(
            l3, l2, 1, "test failed when labels are in (1, -1)"
        )
        check_serialization(1, multiclass_kr)

    def test_hinge_multiclass_loss(self):
        multiclass_hinge = HingeMulticlassLoss(1.0, 1e-7)
        hinge = HingeMarginLoss(1.0, 1e-4)
        # testing with an other value for eps ensure that eps has no influence
        y_true = tf.convert_to_tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        y_pred = tf.convert_to_tensor([0.5, 1.5, -0.5, -0.5, -1.5, 0.5])
        l_single = hinge(y_true, y_pred).numpy()
        l_multi = multiclass_hinge(y_true, y_pred).numpy()
        self.assertEqual(
            l_single,
            l_multi,
            "hinge multiclass must yield the same "
            "results when given a single class "
            "vector",
        )
        n_class = 10
        n_items = 100
        y_true = tf.one_hot(np.random.randint(0, 10, n_items), n_class)
        y_pred = tf.random.normal((n_items, n_class))
        l = multiclass_hinge(y_true, y_pred).numpy()
        l2 = multiclass_hinge(tf.cast(y_true, dtype=tf.int32), y_pred).numpy()
        np.testing.assert_almost_equal(
            l2, l, 1, "test failed when y_true has dtype int32"
        )
        y_true2 = tf.where(y_true == 1.0, 1.0, -1.0)
        l3 = multiclass_hinge(y_true2, y_pred).numpy()
        np.testing.assert_almost_equal(
            l3, l2, 1, "test failed when labels are in (1, -1)"
        )
        check_serialization(1, multiclass_hinge)

    def test_hkr_multiclass_loss(self):
        multiclass_hkr = HKRmulticlassLoss(5, 1.0, 1e-12)
        hkr_binary = HKR_loss(5.0, 1.0)
        # testing with an other value for eps ensure that eps has no influence
        y_true = tf.convert_to_tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        y_pred = tf.convert_to_tensor([0.5, 1.5, -0.5, -0.5, -1.5, 0.5])
        l_single = hkr_binary(y_true, y_pred).numpy()
        l_multi = multiclass_hkr(y_true, y_pred).numpy()
        self.assertEqual(
            l_single,
            l_multi,
            "hinge multiclass must yield the same "
            "results when given a single class "
            "vector",
        )
        n_class = 10
        n_items = 100
        y_true = tf.one_hot(np.random.randint(0, 10, n_items), n_class)
        y_pred = tf.random.normal((n_items, n_class))
        l = multiclass_hkr(y_true, y_pred).numpy()
        l2 = multiclass_hkr(tf.cast(y_true, dtype=tf.int32), y_pred).numpy()
        np.testing.assert_almost_equal(
            l2, l, 1, "test failed when y_true has dtype int32"
        )
        y_true2 = tf.where(y_true == 1.0, 1.0, -1.0)
        l3 = multiclass_hkr(y_true2, y_pred).numpy()
        np.testing.assert_almost_equal(
            l3, l2, 1, "test failed when labels are in (1, -1)"
        )
        check_serialization(1, multiclass_hkr)

    def test_multi_margin_loss(self):
        self.fail()
