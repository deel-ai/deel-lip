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
    TauCategoricalCrossentropy,
    TauSparseCategoricalCrossentropy,
    TauBinaryCrossentropy,
    CategoricalHinge,
)
from deel.lip.utils import process_labels_for_multi_gpu
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


def binary_tf_data(x):
    """Return a TF float32 tensor of shape [N, 1] from a list/np.array of shape [N]"""
    return tf.expand_dims(tf.constant(x, dtype=tf.float32), axis=-1)


class Test(TestCase):
    def test_kr_loss(self):
        loss = KR()

        y_true = binary_tf_data([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        y_pred = binary_tf_data([0.5, 1.5, -0.5, -0.5, -1.5, 0.5])
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
        loss = HingeMargin(2.0)
        y_true = binary_tf_data([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        y_pred = binary_tf_data([0.5, 1.5, -0.5, -0.5, -1.5, 0.5])
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
        multiclass_kr = MulticlassKR(reduction="sum_over_batch_size")
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
        multiclass_hinge = MulticlassHinge(2.0)

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
        multiclass_hkr = MulticlassHKR(5.0, 2.0)
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

    def test_categoricalhinge(self):
        cathinge = CategoricalHinge(1.0)
        n_class = 10
        n_items = 10000
        y_true = tf.one_hot(np.random.randint(0, 10, n_items), n_class)
        y_pred = tf.random.normal((n_items, n_class))
        loss_val = cathinge(y_true, y_pred).numpy()
        loss_val_2 = cathinge(tf.cast(y_true, dtype=tf.int32), y_pred).numpy()
        np.testing.assert_almost_equal(
            loss_val_2, loss_val, 1, "test failed when y_true has dtype int32"
        )
        check_serialization(1, cathinge)

    def test_tau_catcrossent(self):
        taucatcrossent_loss = TauCategoricalCrossentropy(1.0)
        n_class = 10
        n_items = 10000
        y_true = tf.one_hot(np.random.randint(0, 10, n_items), n_class)
        y_pred = tf.random.normal((n_items, n_class))
        loss_val = taucatcrossent_loss(y_true, y_pred).numpy()
        loss_val_2 = taucatcrossent_loss(
            tf.cast(y_true, dtype=tf.int32), y_pred
        ).numpy()
        np.testing.assert_almost_equal(
            loss_val_2, loss_val, 1, "test failed when y_true has dtype int32"
        )
        check_serialization(1, taucatcrossent_loss)

    def test_tau_sparse_catcrossent(self):
        tau_sparse_catcrossent_loss = TauSparseCategoricalCrossentropy(1.0)
        n_class = 10
        n_items = 10000
        y_true = np.random.randint(0, n_class, n_items)
        y_pred = tf.random.normal((n_items, n_class))
        loss_val = tau_sparse_catcrossent_loss(y_true, y_pred).numpy()
        loss_val_2 = tau_sparse_catcrossent_loss(
            tf.cast(y_true, dtype=tf.int32), y_pred
        ).numpy()
        np.testing.assert_almost_equal(
            loss_val_2, loss_val, 1, "test failed when y_true has dtype int32"
        )
        check_serialization(n_class, tau_sparse_catcrossent_loss)

    def test_tau_binary_crossent(self):
        loss = TauBinaryCrossentropy(2.0)
        y_true = binary_tf_data([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        y_pred = binary_tf_data([0.5, 1.5, -0.5, -0.5, -1.5, 0.5])

        # Assert that loss value is equal to expected value
        expected_loss_val = 0.279185
        loss_val = loss(y_true, y_pred).numpy()
        np.testing.assert_allclose(loss_val, expected_loss_val, rtol=1e-6)

        # Assert that loss value is the same when y_true is of type int32
        loss_val_2 = loss(tf.cast(y_true, dtype=tf.int32), y_pred).numpy()
        np.testing.assert_allclose(loss_val_2, expected_loss_val, rtol=1e-6)

        # Assert that loss value is the same when y_true is [-1, 1] instead of [0, 1]
        y_true2 = tf.where(y_true == 1.0, 1.0, -1.0)
        loss_val_3 = loss(y_true2, y_pred).numpy()
        np.testing.assert_allclose(loss_val_3, expected_loss_val, rtol=1e-6)

        # Assert that loss object is correctly serialized
        check_serialization(1, loss)

    def test_no_reduction_binary_losses(self):
        """
        Assert binary losses without reduction. Three losses are tested on hardcoded
        y_true/y_pred of shape [8 elements, 1]: KR, HingeMargin and HKR.
        """
        y_true = binary_tf_data([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        y_pred = binary_tf_data([0.5, 1.1, -0.1, 0.7, -1.3, -0.4, 0.2, -0.9])

        losses = (
            KR(reduction="none"),
            HingeMargin(0.7 * 2.0, reduction="none"),
            HKR(alpha=2.5, min_margin=2.0, reduction="none"),
            TauBinaryCrossentropy(tau=0.5, reduction="none"),
        )

        expected_loss_values = (
            np.array([1.0, 2.2, -0.2, 1.4, 2.6, 0.8, -0.4, 1.8]),
            np.array([0.2, 0, 0.8, 0, 0, 0.3, 0.9, 0]),
            np.array([0.25, -2.2, 2.95, -0.65, -2.6, 0.7, 3.4, -1.55]),
            [1.15188, 0.91098, 1.43692, 1.06676, 0.84011, 1.19628, 1.48879, 0.98650],
        )

        for loss, expected_loss_val in zip(losses, expected_loss_values):
            loss_val = loss(y_true, y_pred)
            np.testing.assert_allclose(
                loss_val,
                expected_loss_val,
                rtol=5e-6,
                atol=5e-6,
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
            MulticlassHinge(min_margin=2.0, reduction="none"),
            MultiMargin(0.7, reduction="none"),
            MulticlassHKR(alpha=2.5, min_margin=1.0, reduction="none"),
            CategoricalHinge(1.1, reduction="none"),
            TauCategoricalCrossentropy(2.0, reduction="none"),
            TauSparseCategoricalCrossentropy(2.0, reduction="none"),
        )

        expected_loss_values = (
            np.float64([326, 314, 120, 250, 496, -258, 396, 450]) / 225,
            np.float64([17, 13, 34, 15, 12, 62, 17, 45]) / 30,
            np.float64([0, 0, 19, 0, 0, 35, 0, 0]) / 30,
            np.float64([-779, -656, 1395, -625, -1609, 4557, -1284, 825]) / 900,
            np.float64([0, 0.4, 2.3, 0.1, 0, 3.9, 0.4, 0]),
            np.float64(
                [
                    0.044275,
                    0.115109,
                    1.243572,
                    0.084923,
                    0.010887,
                    2.802300,
                    0.114224,
                    0.076357,
                ]
            ),
            np.float64(
                [
                    0.044275,
                    0.115109,
                    1.243572,
                    0.084923,
                    0.010887,
                    2.802300,
                    0.114224,
                    0.076357,
                ]
            ),
        )

        for loss, expected_loss_val in zip(losses, expected_loss_values):
            if isinstance(loss, TauSparseCategoricalCrossentropy):
                loss_val = loss(tf.argmax(y_true, axis=-1), y_pred)
            else:
                loss_val = loss(y_true, y_pred)
            np.testing.assert_allclose(
                loss_val,
                expected_loss_val,
                rtol=5e-6,
                atol=5e-6,
                err_msg=f"Loss {loss.name} failed",
            )

    def test_minibatches_binary_losses(self):
        """
        Assert binary losses with mini-batches, to simulate multi-GPU/TPU. Three losses
        are tested using "sum" reduction: KR, HingeMargin and HKR.
        Assertions are tested with:
        - losses on hardcoded small data of size [8, 1] with full batch and mini-batches
        - losses on large random data with full batch and mini-batches.
        """
        # Small hardcoded data of size [8, 1]
        y_true = binary_tf_data([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = binary_tf_data([0.5, 1.1, -0.1, 0.7, -1.3, -0.4, 0.2, -0.9])
        y_true = process_labels_for_multi_gpu(y_true)

        reduction = "sum"
        losses = (
            KR(multi_gpu=True, reduction=reduction),
            HingeMargin(0.7 * 2.0, reduction=reduction),
            HKR(alpha=2.5, min_margin=2.0, multi_gpu=True, reduction=reduction),
            TauBinaryCrossentropy(tau=1.5, reduction=reduction),
        )

        expected_loss_values = (9.2, 2.2, 0.3, 2.19262)

        # Losses are tested on full batch
        for loss, expected_loss_val in zip(losses, expected_loss_values):
            loss_val = loss(y_true, y_pred)
            np.testing.assert_allclose(
                loss_val,
                expected_loss_val,
                rtol=5e-6,
                atol=5e-6,
                err_msg=f"Loss {loss.name} failed for hardcoded full batch",
            )

        # Losses are tested on 3 mini-batches of size [3, 4, 1]
        for loss, expected_loss_val in zip(losses, expected_loss_values):
            loss_val = 0
            loss_val += loss(y_true[:3], y_pred[:3])
            loss_val += loss(y_true[3:7], y_pred[3:7])
            loss_val += loss(y_true[7:], y_pred[7:])

            np.testing.assert_allclose(
                loss_val,
                expected_loss_val,
                rtol=5e-6,
                atol=5e-6,
                err_msg=f"Loss {loss.name} failed for hardcoded mini-batches",
            )

        # Large random data of size [10000, 1]
        num_items = 10000
        y_true = binary_tf_data(np.random.randint(2, size=num_items))
        y_pred = tf.random.normal((num_items, 1))
        y_true = process_labels_for_multi_gpu(y_true)

        # Compare full batch loss and mini-batches loss
        for loss in losses:
            loss_val_full_batch = loss(y_true, y_pred)

            i1 = num_items // 2
            i2 = i1 + num_items // 4
            i3 = i2 + num_items // 6
            loss_val_minibatches = 0
            loss_val_minibatches += loss(y_true[:i1], y_pred[:i1])
            loss_val_minibatches += loss(y_true[i1:i2], y_pred[i1:i2])
            loss_val_minibatches += loss(y_true[i2:i3], y_pred[i2:i3])
            loss_val_minibatches += loss(y_true[i3:], y_pred[i3:])
            np.testing.assert_allclose(
                loss_val_full_batch,
                loss_val_minibatches,
                rtol=5e-6,
                atol=5e-6,
                err_msg=f"Loss {loss.name} failed for random mini-batches",
            )

    def test_minibatches_multiclass_losses(self):
        """
        Assert multiclass losses with mini-batches, to simulate multi-GPU/TPU. Four
        losses are tested using "sum" reduction: MulticlassKR, MulticlassHinge,
        MultiMargin and MulticlassHKR.
        Assertions are tested with:
        - losses on hardcoded small data of size [8, 3] with full batch and mini-batches
        - losses on large random data with full batch and mini-batches.
        """
        # Small hardcoded data of size [8, 3]
        y_true = tf.one_hot([0, 0, 0, 1, 1, 1, 2, 2], 3)
        y_true = process_labels_for_multi_gpu(y_true)
        y_pred = tf.constant(
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
            dtype=tf.float32,
        )

        # Losses are tested on full batch
        reduction = "sum"
        losses = (
            MulticlassKR(multi_gpu=True, reduction=reduction),
            MulticlassHinge(min_margin=2.0, reduction=reduction),
            MultiMargin(0.7, reduction=reduction),
            MulticlassHKR(
                alpha=2.5, min_margin=1.0, multi_gpu=True, reduction=reduction
            ),
        )

        expected_loss_values = (
            np.float32(698 / 75),
            np.float32(43 / 6),
            np.float32(9 / 5),
            np.float32(152 / 75),
        )

        # Losses are tested on 3 mini-batches of size [3, 4, 1]
        for loss, expected_loss_val in zip(losses, expected_loss_values):
            loss_val = loss(y_true, y_pred)
            np.testing.assert_allclose(
                loss_val,
                expected_loss_val,
                rtol=5e-6,
                atol=5e-6,
                err_msg=f"Loss {loss.name} failed for hardcoded full batch",
            )

        # The dataset is now split into 3 mini-batches of size [3, 4, 1]
        for loss, expected_loss_val in zip(losses, expected_loss_values):
            loss_val = 0
            loss_val += loss(y_true[:3], y_pred[:3])
            loss_val += loss(y_true[3:7], y_pred[3:7])
            loss_val += loss(y_true[7:], y_pred[7:])

            np.testing.assert_allclose(
                loss_val,
                expected_loss_val,
                rtol=5e-6,
                atol=5e-6,
                err_msg=f"Loss {loss.name} failed for hardcoded mini-batches",
            )

        # Large random data of size [10000, 1]
        num_classes = 10
        num_items = 10000
        y_true = tf.one_hot(np.random.randint(num_classes, size=num_items), num_classes)
        y_true = process_labels_for_multi_gpu(y_true)
        y_pred = tf.random.normal((num_items, num_classes), seed=17)

        # Compare full batch loss and mini-batches loss
        for loss in losses:
            loss_val_full_batch = loss(y_true, y_pred)

            i1 = num_items // 2
            i2 = i1 + num_items // 4
            i3 = i2 + num_items // 6
            loss_val_minibatches = 0
            loss_val_minibatches += loss(y_true[:i1], y_pred[:i1])
            loss_val_minibatches += loss(y_true[i1:i2], y_pred[i1:i2])
            loss_val_minibatches += loss(y_true[i2:i3], y_pred[i2:i3])
            loss_val_minibatches += loss(y_true[i3:], y_pred[i3:])
            np.testing.assert_allclose(
                loss_val_full_batch,
                loss_val_minibatches,
                rtol=5e-6,
                atol=5e-6,
                err_msg=f"Loss {loss.name} failed for random mini-batches",
            )

    def test_multilabel_losses(self):
        """
        Assert binary losses with multilabels.
        Three losses are tested (KR, HingeMargin and HKR). We compare losses with three
        separate binary classification and the corresponding multilabel problem.
        """
        # Create predictions and labels for 3 binary problems and the concatenated
        # multilabel one.
        y_pred1, y_true1 = get_gaussian_data(1000)
        y_pred2, y_true2 = get_gaussian_data(1000)
        y_pred3, y_true3 = get_gaussian_data(1000)
        y_pred = tf.concat([y_pred1, y_pred2, y_pred3], axis=-1)
        y_true = tf.concat([y_true1, y_true2, y_true3], axis=-1)

        # Tested losses (different reductions, multi_gpu)
        losses = (
            KR(reduction="none", name="KR none"),
            HingeMargin(0.4, reduction="none", name="hinge none"),
            HKR(alpha=5.2, reduction="none", name="HKR none"),
            KR(reduction="sum_over_batch_size", name="KR sum_over_bs"),
            HingeMargin(0.6, reduction="sum_over_batch_size", name="hinge sum_over_bs"),
            HKR(alpha=10, reduction="sum_over_batch_size", name="HKR sum_over_bs"),
            KR(multi_gpu=True, reduction="sum", name="KR multi_gpu"),
            HKR(alpha=3.2, multi_gpu=True, reduction="sum", name="HKR multi_gpu"),
        )

        # Compute loss values and assert that the multilabel value is equal to the mean
        # of the three separate binary problems.
        for loss in losses:
            loss_val1 = loss(y_true1, y_pred1).numpy()
            loss_val2 = loss(y_true2, y_pred2).numpy()
            loss_val3 = loss(y_true3, y_pred3).numpy()
            mean_loss_vals = (loss_val1 + loss_val2 + loss_val3) / 3

            loss_val_multilabel = loss(y_true, y_pred).numpy()

            np.testing.assert_allclose(
                loss_val_multilabel,
                mean_loss_vals,
                rtol=5e-6,
                atol=5e-6,
                err_msg=f"Loss {loss.name} failed",
            )
