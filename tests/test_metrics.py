import unittest
from collections import defaultdict
from unittest import TestCase
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD
from deel.lip.metrics import (
    ProvableRobustAccuracy,
    ProvableAvgRobustness,
)
import numpy as np
import os


def check_serialization(nb_class, loss, nb_classes):
    n = 255
    x = tf.random.uniform((n, 42))
    y = tf.one_hot(
        tf.convert_to_tensor(np.random.randint(nb_classes, size=n)), nb_classes
    )
    m = Sequential([Input(42), Dense(nb_class)])
    m.compile(optimizer=SGD(), loss=loss)
    l1 = m.evaluate(x, y)
    name = loss.__class__.__name__
    path = os.path.join("logs", "losses", name)
    m.save(path)
    m2 = load_model(path, compile=True)
    l2 = m2.evaluate(x, y)
    return l1, l2


class Test(TestCase):
    def test_serialization(self):
        pra = ProvableRobustAccuracy(1, 1.0, disjoint_neurons=False)
        l1, l2 = check_serialization(10, pra, 10)
        self.assertEqual(
            l1, l2, "serialization changed loss value for ProvableRobustAccuracy"
        )
        par = ProvableAvgRobustness(
            2.0, disjoint_neurons=False, negative_robustness=True
        )
        l1, l2 = check_serialization(10, par, 10)
        self.assertEqual(
            l1, l2, "serialization changed loss value for ProvableAvgRobustness"
        )

    def test_provable_vs_adjusted(self):
        n = 500
        x = tf.random.normal((n, 10), mean=0.0, stddev=0.1)
        y = tf.one_hot(tf.convert_to_tensor(np.random.randint(10, size=n)), 10)
        for lip_cst in [1, 0.5, 2.0]:
            for disjoint in [True, False]:
                pr = ProvableAvgRobustness(lip_cst, disjoint, negative_robustness=False)
                ar = ProvableAvgRobustness(lip_cst, disjoint, negative_robustness=True)
                l1 = pr(y, y).numpy()
                l2 = ar(y, y).numpy()
                self.assertAlmostEqual(
                    l1,
                    l2,
                    4,
                    msg="provable and adjusted "
                    "robustness must give same "
                    "values when y_true==y_pred",
                )
                l1 = pr(y, x).numpy()
                l2 = ar(y, x).numpy()
                self.assertNotAlmostEqual(
                    l1,
                    l2,
                    4,
                    msg="provable and adjusted robustness must give different values"
                    " when y_true!=y_pred",
                )

        x = tf.random.normal((n, 10), mean=0.0, stddev=0.1)
        y = tf.one_hot(tf.convert_to_tensor(np.random.randint(10, size=n)), 10)
        for lip_cst in [1.0, 0.5, 2.0]:
            bpr = ProvableAvgRobustness(lip_cst, disjoint, negative_robustness=False)
            bar = ProvableAvgRobustness(lip_cst, disjoint, negative_robustness=True)
            l1 = bpr(y, y).numpy()
            l2 = bar(y, y).numpy()
            self.assertAlmostEqual(
                l1,
                l2,
                4,
                msg="provable and adjusted robustness must give same values when y_"
                "true==y_pred",
            )
            l1 = pr(y, x).numpy()
            l2 = ar(y, x).numpy()
            self.assertNotAlmostEqual(
                l1,
                l2,
                4,
                msg="provable and adjusted robustness must give different values when "
                "y_true!=y_pred",
            )

    def test_data_format(self):
        pr = ProvableAvgRobustness(1.0, True, negative_robustness=False)
        ar = ProvableAvgRobustness(1.0, True, negative_robustness=True)
        n = 500
        # check in multiclass
        for metric in [pr, ar]:
            y_pred = tf.random.normal((n, 10), mean=0.0, stddev=0.1)
            y_true = tf.one_hot(tf.convert_to_tensor(np.random.randint(10, size=n)), 10)
            metrics_values = []
            for neg_val in [0.0, -1.0]:
                y_pred_case = y_pred
                y_true_case = tf.where(y_true > 0, 1.0, neg_val)
                metrics_values.append(metric(y_true_case, y_pred_case).numpy())
            self.assertTrue(
                all([m == metrics_values[0] for m in metrics_values]),
                "changing the data format must not change the metric value",
            )
        # check in binary
        for metric in [pr, ar]:
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

    def test_disjoint_neurons(self):
        pr = ProvableAvgRobustness(1.0, False, negative_robustness=False)
        ar = ProvableAvgRobustness(1.0, False, negative_robustness=True)
        pdr = ProvableAvgRobustness(1.0, True, negative_robustness=False)
        adr = ProvableAvgRobustness(1.0, True, negative_robustness=True)
        n = 500
        # check in multiclass
        metrics_values = defaultdict(list)
        y_pred = tf.random.normal((n, 10), mean=0.0, stddev=0.1)
        y_true = tf.one_hot(tf.convert_to_tensor(np.random.randint(10, size=n)), 10)
        for neg_val in [0.0, -1.0]:
            y_pred_case = y_pred
            y_true_case = tf.where(y_true > 0, 1.0, neg_val)
            metrics_values["standard"].append(pdr(y_true_case, y_pred_case).numpy())
            metrics_values["standard"].append(
                pr(y_true_case, y_pred_case * (np.sqrt(2) / 2.0)).numpy()
            )
            metrics_values["negative_rob"].append(adr(y_true_case, y_pred_case).numpy())
            metrics_values["negative_rob"].append(
                ar(y_true_case, y_pred_case * (np.sqrt(2) / 2.0)).numpy()
            )
        self.assertTrue(
            all(
                [m == metrics_values["standard"][0] for m in metrics_values["standard"]]
            ),
            "the loss does not account disjoint_neuron properly in multiclass",
        )
        self.assertTrue(
            all(
                [
                    m == metrics_values["negative_rob"][0]
                    for m in metrics_values["negative_rob"]
                ]
            ),
            "the loss does not account disjoint_neuron properly in multiclass",
        )
        # check in binary
        metrics_values = defaultdict(list)
        y_pred = tf.random.normal((n,), mean=0.0, stddev=0.1)
        y_true = tf.random.normal((n,), mean=0.0, stddev=0.1)
        for neg_val in [0.0, -1.0]:
            y_pred_case = y_pred
            y_true_case = tf.where(y_true > 0, 1.0, neg_val)
            # no sqrt(2)/2 here as the corrective factor works only in multiclass
            metrics_values["standard"].append(pdr(y_true_case, y_pred_case).numpy())
            metrics_values["standard"].append(pr(y_true_case, y_pred_case).numpy())
            metrics_values["negative_rob"].append(adr(y_true_case, y_pred_case).numpy())
            metrics_values["negative_rob"].append(ar(y_true_case, y_pred_case).numpy())
        self.assertTrue(
            all(
                [m == metrics_values["standard"][0] for m in metrics_values["standard"]]
            ),
            "the loss does not account disjoint_neuron properly in binary",
        )
        self.assertTrue(
            all(
                [
                    m == metrics_values["negative_rob"][0]
                    for m in metrics_values["negative_rob"]
                ]
            ),
            "the loss does not account disjoint_neuron properly in binary",
        )

    def test_hardcoded_values(self):
        pr = ProvableRobustAccuracy(1.0, False)
        ar = ProvableAvgRobustness(2.0, False)


if __name__ == "__main__":
    unittest.main()
