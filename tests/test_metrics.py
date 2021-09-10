import unittest
from unittest import TestCase
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD
from deel.lip.utils import load_model
from deel.lip.metrics import (
    ProvableRobustness,
    AdjustedRobustness,
    BinaryAdjustedRobustness,
    BinaryProvableRobustness,
)
import os


def check_serialization(nb_class, loss):
    m = Sequential([Input(10), Dense(nb_class)])
    m.compile(optimizer=SGD(), loss=loss)
    name = loss.__class__.__name__
    path = os.path.join("logs", "losses", name)
    m.save(path)
    m2 = load_model(path, compile=True)
    m2(tf.random.uniform((255, 10)))


class Test(TestCase):
    def test_serialization(self):
        pr = ProvableRobustness(1.0, False)
        check_serialization(1, pr)
        ar = AdjustedRobustness(2.0, False)
        check_serialization(1, ar)
        bar = BinaryAdjustedRobustness(1.0)
        check_serialization(1, bar)
        bpr = BinaryProvableRobustness(1.0)
        check_serialization(1, bpr)

    def test_provable_vs_adjusted(self):
        n = 500
        x1 = tf.random.normal((n, 10), mean=0.0, stddev=0.1)
        x2 = tf.random.normal((n, 10), mean=1.0, stddev=0.1)
        for l in [1.0, 0.5, 2.0]:
            for disjoint in [True, False]:
                pr = ProvableRobustness(l, disjoint)
                ar = AdjustedRobustness(l, disjoint)
                l1 = pr(x1, x1).numpy()
                l2 = ar(x1, x1).numpy()
                self.assertAlmostEqual(
                    l1,
                    l2,
                    4,
                    msg="provable and adjusted "
                    "robustness must give same "
                    "values when y_true==y_pred",
                )
                l1 = pr(x1, x2).numpy()
                l2 = ar(x1, x2).numpy()
                self.assertNotAlmostEqual(
                    l1,
                    l2,
                    4,
                    msg="provable and adjusted robustness must give different values"
                    " when y_true!=y_pred",
                )

        x1 = tf.random.normal((n, 10), mean=0.0, stddev=0.1)
        x2 = tf.random.normal((n, 10), mean=1.0, stddev=0.1)
        for l in [1.0, 0.5, 2.0]:
            bar = BinaryAdjustedRobustness(l)
            bpr = BinaryProvableRobustness(l)
            l1 = bpr(x1, x1).numpy()
            l2 = bar(x1, x1).numpy()
            self.assertAlmostEqual(
                l1,
                l2,
                4,
                msg="provable and adjusted robustness must give same values when y_"
                "true==y_pred",
            )
            l1 = bpr(x1, x2).numpy()
            l2 = bar(x1, x2).numpy()
            self.assertNotAlmostEqual(
                l1,
                l2,
                4,
                msg="provable and adjusted robustness must give different values when "
                "y_true!=y_pred",
            )

    def test_data_format(self):
        pr = ProvableRobustness(1.0, False)
        ar = AdjustedRobustness(2.0, False)
        bar = BinaryAdjustedRobustness(1.0)
        bpr = BinaryProvableRobustness(1.0)
        n = 500
        for metric in [pr, ar]:
            y_pred = tf.random.normal((n, 10), mean=0.0, stddev=0.1)
            y_true = tf.random.normal((n, 10), mean=0.0, stddev=0.1)
            metrics_values = []
            for neg_val in [0.0, -1.0]:
                y_pred_case = y_pred
                y_true_case = tf.where(y_true > 0, 1.0, neg_val)
                metrics_values.append(metric(y_true_case, y_pred_case).numpy())
            self.assertTrue(
                all([m == metrics_values[0] for m in metrics_values]),
                "changing the data format must not change the metric " "value",
            )
        for metric in [bar, bpr]:
            y_pred = tf.random.normal((n,), mean=0.0, stddev=0.1)
            y_true = tf.random.normal((n,), mean=0.0, stddev=0.1)
            metrics_values = []
            for expand_dim in [True, False]:
                for neg_val in [0.0, -1.0]:
                    y_pred_case = y_pred
                    y_true_case = tf.where(y_true > 0, 1.0, neg_val)
                    if expand_dim:
                        y_true_case = tf.expand_dims(y_true_case, -1)
                        y_pred_case = tf.expand_dims(y_pred_case, -1)
                    metrics_values.append(metric(y_true_case, y_pred_case).numpy())
            self.assertTrue(
                all([m == metrics_values[0] for m in metrics_values]),
                "changing the data format must not change the metric " "value",
            )


if __name__ == "__main__":
    unittest.main()
