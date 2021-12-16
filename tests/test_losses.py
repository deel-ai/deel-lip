from unittest import TestCase
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from deel.lip.losses import (
    KR,
    HingeMargin,
    HKR,
    MulticlassKR,
    MulticlassHinge,
    MulticlassHKR,
    MultiMargin,
)
import os
import numpy as np

# a tester:
# - un cas hardcodé
# - les dtypes pour y_true
# - shape y_pred
# - labels en [0,1] et en [-1,1]
# - liens entre les losses


def get_gaussian_data(n=500, mean1=1.0, mean2=-1.0):
    x1 = tf.random.normal((n, 1), mean=mean1, stddev=0.1)
    x2 = tf.random.normal((n, 1), mean=mean2, stddev=0.1)
    y_pred = tf.concat([x1, x2], axis=0)
    y_true = tf.concat([tf.ones((n, 1)), tf.zeros((n, 1))], axis=0)
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
        loss = KR()

        y_true = tf.convert_to_tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        y_pred = tf.convert_to_tensor([0.5, 1.5, -0.5, -0.5, -1.5, 0.5])
        loss_val = loss(y_true, y_pred).numpy()
        self.assertEqual(loss_val, np.float32(1), "KR loss must be equal to 1")

        y_pred, y_true = get_gaussian_data(20000)
        loss_val = loss(y_true, y_pred).numpy()
        np.testing.assert_approx_equal(
            loss_val, 2.0, 1, "test failed when y_true has shape (bs, 1)"
        )
        loss_val_3 = loss(tf.cast(y_true, dtype=tf.int32), y_pred).numpy()
        self.assertEqual(
            loss_val_3, loss_val, "test failed when y_true has dtype int32"
        )
        y_true2 = tf.where(y_true == 1.0, 1.0, -1.0)
        loss_val_4 = loss(y_true2, y_pred).numpy()
        self.assertEqual(loss_val_4, loss_val, "test failed when labels are in (1, -1)")

        check_serialization(1, loss)

    def test_hkr_loss(self):
        # assuming KR and hinge have been properly tested
        loss = HKR(0.5, 2.0)
        check_serialization(1, loss)

    def test_hinge_margin_loss(self):
        loss = HingeMargin(1.0)
        y_true = tf.convert_to_tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        y_pred = tf.convert_to_tensor([0.5, 1.5, -0.5, -0.5, -1.5, 0.5])
        loss_val = loss(y_true, y_pred).numpy()
        self.assertEqual(loss_val, np.float32(4 / 6), "Hinge loss must be equal to 4/6")
        loss_val_3 = loss(tf.cast(y_true, dtype=tf.int32), y_pred).numpy()
        self.assertEqual(
            loss_val_3, np.float32(4 / 6), "Hinge loss must be equal to 4/6"
        )
        y_true2 = tf.where(y_true == 1.0, 1.0, -1.0)
        loss_val_4 = loss(y_true2, y_pred).numpy()
        self.assertEqual(
            loss_val_4, np.float32(4 / 6), "Hinge loss must be equal to 4/6"
        )
        check_serialization(1, loss)

    def test_kr_multiclass_loss(self):
        multiclass_kr = MulticlassKR(reduction="auto")
        y_true = tf.one_hot([0, 0, 0, 1, 1, 2], 3)
        y_pred = np.float32(
            [
                [2, 0.2, -0.5],
                [-1, -1.2, 0.3],
                [0.8, 2, 0],
                [0, 1, -0.5],
                [2.4, -0.4, -1.1],
                [-0.1, -1.7, 0.6],
            ]
        )
        loss_val = multiclass_kr(y_true, y_pred).numpy()
        np.testing.assert_allclose(loss_val, np.float32(761 / 1800))

        n_class = 10
        n_items = 10000
        y_true = tf.one_hot(np.random.randint(0, 10, n_items), n_class)
        y_pred = tf.random.normal((n_items, n_class))
        loss_val = multiclass_kr(y_true, y_pred).numpy()
        loss_val_2 = multiclass_kr(tf.cast(y_true, dtype=tf.int32), y_pred).numpy()
        self.assertEqual(
            loss_val_2, loss_val, "test failed when y_true has dtype int32"
        )
        check_serialization(1, multiclass_kr)

    def test_hinge_multiclass_loss(self):
        multiclass_hinge = MulticlassHinge(1.0)
        hinge = HingeMargin(1.0)
        y_true = tf.convert_to_tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        y_pred = tf.convert_to_tensor([0.5, 1.5, -0.5, -0.5, -1.5, 0.5])
        l_single = hinge(y_true, y_pred).numpy()
        l_multi = multiclass_hinge(y_true, y_pred).numpy()
        self.assertEqual(l_single, np.float32(4 / 6), "Hinge loss must be equal to 4/6")
        self.assertEqual(
            l_single,
            l_multi,
            "hinge multiclass must yield the same "
            "results when given a single class "
            "vector",
        )
        y_true = tf.expand_dims(y_true, -1)
        y_pred = tf.expand_dims(y_pred, -1)
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
        n_items = 10000
        y_true = tf.one_hot(np.random.randint(0, 10, n_items), n_class)
        y_pred = tf.random.normal((n_items, n_class))
        loss_val = multiclass_hinge(y_true, y_pred).numpy()
        loss_val_2 = multiclass_hinge(tf.cast(y_true, dtype=tf.int32), y_pred).numpy()
        self.assertEqual(
            loss_val_2, loss_val, "test failed when y_true has dtype int32"
        )
        y_true2 = tf.where(y_true == 1.0, 1.0, -1.0)
        loss_val_3 = multiclass_hinge(y_true2, y_pred).numpy()
        self.assertEqual(
            loss_val_3, loss_val_2, "test failed when labels are in (1, -1)"
        )
        check_serialization(1, multiclass_hinge)

    def test_hkr_multiclass_loss(self):
        multiclass_hkr = MulticlassHKR(5.0, 1.0)
        y_true = tf.one_hot([0, 0, 0, 1, 1, 2], 3)
        y_pred = np.float32(
            [
                [2, 0.2, -0.5],
                [-1, -1.2, 0.3],
                [0.8, 2, 0],
                [0, 1, -0.5],
                [2.4, -0.4, -1.1],
                [-0.1, -1.7, 0.6],
            ]
        )
        loss_val = multiclass_hkr(y_true, y_pred).numpy()
        np.testing.assert_allclose(loss_val, np.float32(1071 / 200))

        n_class = 10
        n_items = 10000
        y_true = tf.one_hot(np.random.randint(0, 10, n_items), n_class)
        y_pred = tf.random.normal((n_items, n_class))
        loss_val = multiclass_hkr(y_true, y_pred).numpy()
        loss_val_2 = multiclass_hkr(tf.cast(y_true, dtype=tf.int32), y_pred).numpy()
        self.assertEqual(
            loss_val_2,
            loss_val,
            "test failed when y_true has dtype int32",
        )
        check_serialization(1, multiclass_hkr)

    def test_multi_margin_loss(self):
        multimargin_loss = MultiMargin(1.0)
        n_class = 10
        n_items = 10000
        y_true = tf.one_hot(np.random.randint(0, 10, n_items), n_class)
        y_pred = tf.random.normal((n_items, n_class))
        loss_val = multimargin_loss(y_true, y_pred).numpy()
        loss_val_2 = multimargin_loss(tf.cast(y_true, dtype=tf.int32), y_pred).numpy()
        self.assertEqual(
            loss_val_2, loss_val, "test failed when y_true has dtype int32"
        )
        y_true2 = tf.where(y_true == 1.0, 1.0, -1.0)
        loss_val_3 = multimargin_loss(y_true2, y_pred).numpy()
        self.assertEqual(
            loss_val_3, loss_val_2, "test failed when labels are in (1, -1)"
        )
        check_serialization(1, multimargin_loss)

    def test_no_reduction_binary_losses(self):
        """
        Assert binary losses without reduction. Three losses are tested on hardcoded
        y_true/y_pred of shape [8 elements, 1]: KR, HingeMargin and HKR.
        """
        y_true = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]).reshape((8, 1))
        y_pred = np.array([0.5, 1.1, -0.1, 0.7, -1.3, -0.4, 0.2, -0.9]).reshape((8, 1))

        losses = (
            KR(reduction="none"),
            HingeMargin(0.7, reduction="none"),
            HKR(alpha=2.5, reduction="none"),
        )

        expected_loss_values = (
            np.array([1.0, 2.2, -0.2, 1.4, 2.6, 0.8, -0.4, 1.8]),
            np.array([0.2, 0, 0.8, 0, 0, 0.3, 0.9, 0]),
            np.array([0.25, -2.2, 2.95, -0.65, -2.6, 0.7, 3.4, -1.55]),
        )

        for loss, expected_loss_val in zip(losses, expected_loss_values):
            loss_val = loss(y_true, y_pred)
            np.testing.assert_allclose(
                loss_val,
                expected_loss_val,
                rtol=2e-7,
                err_msg=f"Loss {loss.name} failed",
            )

    def test_no_reduction_multiclass_losses(self):
        """
        Assert multi-class losses without reduction. Four losses are tested on hardcoded
        y_true/y_pred of shape [8 elements, 3 classes]: MulticlassKR, MulticlassHinge,
        MultiMargin and MulticlassHKR.
        """
        y_true = tf.one_hot([0, 0, 0, 1, 1, 1, 2, 2], 3)
        y_pred = np.array(
            [
                [1.5, 0.2, -0.5],
                [1, -1.2, 0.3],
                [0.8, 2, -2],
                [0, 1, -0.5],
                [-0.8, 2, 0],
                [2.4, -0.4, -1.1],
                [-0.1, -1.7, 0.6],
                [1.2, 1.3, 2.5],
            ],
            dtype=np.float32,
        )
        losses = (
            MulticlassKR(reduction="none"),
            MulticlassHinge(reduction="none"),
            MultiMargin(0.7, reduction="none"),
            MulticlassHKR(alpha=2.5, min_margin=0.5, reduction="none"),
        )

        expected_loss_values = (
            np.float64([326, 314, 120, 250, 496, -258, 396, 450]) / 225,
            np.float64([17, 13, 34, 15, 12, 62, 17, 45]) / 30,
            np.float64([0, 0, 19, 0, 0, 35, 0, 0]) / 30,
            np.float64([-779, -656, 1395, -625, -1609, 4557, -1284, 825]) / 900,
        )

        for loss, expected_loss_val in zip(losses, expected_loss_values):
            loss_val = loss(y_true, y_pred)
            np.testing.assert_allclose(
                loss_val,
                expected_loss_val,
                rtol=5e-7,
                err_msg=f"Loss {loss.name} failed",
            )
