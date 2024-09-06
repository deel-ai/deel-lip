import unittest
from collections import defaultdict
from unittest import TestCase
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD
from deel.lip.metrics import (
    CategoricalProvableRobustAccuracy,
    BinaryProvableRobustAccuracy,
    CategoricalProvableAvgRobustness,
    BinaryProvableAvgRobustness,
)
import numpy as np
import os


def check_serialization(nb_class, loss, nb_classes):
    n = 255
    x = tf.random.uniform((n, 42))
    y = tf.one_hot(
        tf.convert_to_tensor(np.random.randint(nb_classes, size=n)), nb_classes
    )
    m = Sequential([Input((42,)), Dense(nb_class)])
    m.compile(optimizer=SGD(), loss=loss)
    l1 = m.evaluate(x, y)
    name = loss.__class__.__name__ + ".keras"
    path = os.path.join("logs", "losses", name)
    m.save(path)
    m2 = load_model(path, compile=True)
    l2 = m2.evaluate(x, y)
    return l1, l2


class Test(TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs("logs/lip_layers", exist_ok=True)

    def test_serialization(self):
        pra = CategoricalProvableRobustAccuracy(1, 1.0, disjoint_neurons=False)
        l1, l2 = check_serialization(10, pra, 10)
        self.assertEqual(
            l1,
            l2,
            "serialization changed loss value for CategoricalProvableRobustAccuracy",
        )
        pra = BinaryProvableRobustAccuracy(1, 1.0)
        l1, l2 = check_serialization(10, pra, 10)
        self.assertEqual(
            l1, l2, "serialization changed loss value for BinaryProvableRobustAccuracy"
        )
        par = CategoricalProvableAvgRobustness(
            2.0, disjoint_neurons=False, negative_robustness=True
        )
        l1, l2 = check_serialization(10, par, 10)
        self.assertEqual(
            l1,
            l2,
            "serialization changed loss value for CategoricalProvableAvgRobustness",
        )
        par = BinaryProvableAvgRobustness(2.0, negative_robustness=True)
        l1, l2 = check_serialization(10, par, 10)
        self.assertEqual(
            l1, l2, "serialization changed loss value for BinaryProvableAvgRobustness"
        )

    def test_provable_vs_adjusted(self):
        n = 500
        x = tf.random.normal((n, 10), mean=0.0, stddev=0.1)
        y = tf.one_hot(tf.convert_to_tensor(np.random.randint(10, size=n)), 10)
        for lip_cst in [1, 0.5, 2.0]:
            for disjoint in [True, False]:
                pr = CategoricalProvableAvgRobustness(
                    lip_cst, disjoint, negative_robustness=False
                )
                ar = CategoricalProvableAvgRobustness(
                    lip_cst, disjoint, negative_robustness=True
                )
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
            bpr = BinaryProvableAvgRobustness(lip_cst, negative_robustness=False)
            bar = BinaryProvableAvgRobustness(lip_cst, negative_robustness=True)
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
        pr = CategoricalProvableAvgRobustness(1.0, True, negative_robustness=False)
        ar = CategoricalProvableAvgRobustness(1.0, True, negative_robustness=True)
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
        bpr = BinaryProvableAvgRobustness(1.0, negative_robustness=False)
        bar = BinaryProvableAvgRobustness(1.0, negative_robustness=True)
        for metric in [bpr, bar]:
            y_pred = tf.random.normal((n,), mean=0.0, stddev=0.1)
            y_true = tf.random.normal((n,), mean=0.0, stddev=0.1)
            metrics_values = []
            for neg_val in [0.0, -1.0]:
                y_pred_case = y_pred
                y_true_case = tf.where(y_true > 0, 1.0, neg_val)
                metrics_values.append(metric(y_true_case, y_pred_case).numpy())
            self.assertTrue(
                all([m == metrics_values[0] for m in metrics_values]),
                "changing the data format must not change the metric value",
            )

    def test_disjoint_neurons(self):
        pr = CategoricalProvableAvgRobustness(1.0, False, negative_robustness=False)
        ar = CategoricalProvableAvgRobustness(1.0, False, negative_robustness=True)
        pdr = CategoricalProvableAvgRobustness(1.0, True, negative_robustness=False)
        adr = CategoricalProvableAvgRobustness(1.0, True, negative_robustness=True)
        n = 500
        # check in multiclass
        metrics_values = defaultdict(list)
        y_pred = tf.random.normal((n, 10), mean=0.0, stddev=1.0)
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
        print(metrics_values)
        for metric_type in ["standard", "negative_rob"]:
            for m in metrics_values[metric_type]:
                self.assertAlmostEqual(
                    m,
                    metrics_values[metric_type][0],
                    4,
                    "the loss does not account disjoint_neuron properly in multiclass",
                )
        # check in binary
        pr = BinaryProvableAvgRobustness(1.0, negative_robustness=False)
        ar = BinaryProvableAvgRobustness(1.0, negative_robustness=True)
        metrics_values = defaultdict(list)
        y_pred = tf.random.normal((n,), mean=0.0, stddev=0.1)
        y_true = tf.random.normal((n,), mean=0.0, stddev=0.1)
        for neg_val in [0.0, -1.0]:
            y_pred_case = y_pred
            y_true_case = tf.where(y_true > 0, 1.0, neg_val)
            # no sqrt(2)/2 here as the corrective factor works only in multiclass
            metrics_values["standard"].append(pr(y_true_case, y_pred_case).numpy())
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
        pr = CategoricalProvableRobustAccuracy(0.25, 1.0, False)
        ar = CategoricalProvableAvgRobustness(1.0, False)
        y_pred = tf.convert_to_tensor(
            [
                [1.0, 0.0, 0.0],  # good class & over 0.25
                [0.1, 0.0, 0.0],  # good class & below 0.25
                [0.0, 0.0, 1.0],  # wrong class & over 0.25
                [0.0, 0.0, 0.1],  # wrong class & below 0.25
            ]
        )
        y_true = tf.convert_to_tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        self.assertEqual(
            pr(y_true, y_pred).numpy(),
            0.25,
            "ProvableRobustAccuracy did not "
            "get the expected result with "
            "hardcoded values ",
        )
        self.assertAlmostEqual(
            ar(y_true, y_pred).numpy(),
            0.25 * 1.1 / np.sqrt(2),
            5,
            "ProvableAvgRobustness did not "
            "get the expected result with "
            "hardcoded values ",
        )
        bpr = BinaryProvableRobustAccuracy(0.25, 1.0)
        bar = BinaryProvableAvgRobustness(1.0, negative_robustness=False)
        y_pred = tf.convert_to_tensor(
            [
                1.0,  # good class & over 0.25
                0.1,  # good class & below 0.25
                1.0,  # wrong class & over 0.25
                0.1,  # wrong class & below 0.25
                -1.0,
                -0.1,
                -1.0,
                -0.1,
            ]
        )
        y_true = tf.convert_to_tensor([1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0])
        self.assertEqual(
            bpr(y_true, y_pred).numpy(),
            0.25,
            "BinaryProvableRobustAccuracy did not "
            "get the expected result with "
            "hardcoded values ",
        )
        self.assertAlmostEqual(
            bar(y_true, y_pred).numpy(),
            0.125 * (1.1 * 2),
            5,
            "BinaryProvableAvgRobustness did not "
            "get the expected result with "
            "hardcoded values ",
        )


if __name__ == "__main__":
    unittest.main()
