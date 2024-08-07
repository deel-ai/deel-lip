from unittest import TestCase
import keras
import keras.ops as K
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.losses import CategoricalCrossentropy
from deel.lip.activations import GroupSort, Householder
import os
import numpy as np


def check_serialization(layer):
    m = Sequential([keras.Input((10,)), layer])
    m.compile(optimizer=SGD(), loss=CategoricalCrossentropy(from_logits=False))
    name = layer.__class__.__name__ + ".keras"
    path = os.path.join("logs", "losses", name)
    x = keras.random.uniform((255, 10), -10, 10)
    y1 = m(x)
    m.save(path)
    m2 = load_model(path, compile=True)
    y2 = m2(x)
    np.testing.assert_allclose(y1.numpy(), y2.numpy())


class TestGroupSort(TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs("logs/losses", exist_ok=True)

    def test_simple(self):
        gs = GroupSort(n=2)
        check_serialization(gs)
        gs = GroupSort(n=5)
        check_serialization(gs)

    def test_flat_input(self):
        gs = GroupSort(n=2)
        x = K.convert_to_tensor(
            [
                [1, 2, 3, 4],
                [4, 3, 2, 1],
                [1, 2, 1, 2],
                [2, 1, 2, 1],
            ],
            dtype="float32",
        )
        gs.build((None, 4))
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
        x = K.convert_to_tensor(
            [
                [1, 2, 3, 4],
                [4, 3, 2, 1],
                [1, 2, 1, 2],
                [2, 1, 2, 1],
            ],
            dtype="float32",
        )
        gs.build((None, 4))
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
        x = K.convert_to_tensor(
            [
                [1, 2, 3, 4],
                [4, 3, 2, 1],
                [1, 2, 1, 2],
                [2, 1, 2, 1],
            ],
            dtype="float32",
        )
        x = K.repeat(K.expand_dims(K.repeat(K.expand_dims(x, 1), 28, 1), 1), 28, 1)
        gs.build((None, 28, 28, 4))
        y = gs(x).numpy()
        y_t = [
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 4.0, 1.0, 2.0],
            [1.0, 2.0, 1.0, 2.0],
            [1.0, 2.0, 1.0, 2.0],
        ]
        y_t = K.repeat(K.expand_dims(K.repeat(K.expand_dims(y_t, 1), 28, 1), 1), 28, 1)
        np.testing.assert_equal(y, y_t)

    def test_idempotence(self):
        gs = GroupSort(n=2)
        x = keras.random.uniform((255, 16), -10, 10)
        gs.build((None, 16))
        y1 = gs(x)
        y2 = gs(y1)
        np.testing.assert_equal(y1.numpy(), y2.numpy())
        gs = GroupSort(n=2)
        x = keras.random.uniform((255, 16), -10, 10)
        gs.build((None, 16))
        y1 = gs(x)
        y2 = gs(y1)
        np.testing.assert_equal(y1.numpy(), y2.numpy())


class TestHouseholder(TestCase):
    """Tests for Householder activation:
    - instantiation of layer
    - check outputs on dense (bs, n) tensor, with three thetas: 0, pi/2 and pi
    - check outputs on dense (bs, h, w, n) tensor, with three thetas: 0, pi/2 and pi
    - check idempotence hh(hh(x)) = hh(x)
    """

    @classmethod
    def setUpClass(cls):
        os.makedirs("logs/losses", exist_ok=True)

    def test_instantiation(self):
        # Instantiation without argument
        hh = Householder()
        hh.build((28, 28, 10))
        assert hh.theta.shape == (5,)
        np.testing.assert_equal(hh.theta.numpy(), np.pi / 2)

        # Instantiation with arguments
        hh = Householder(
            data_format="channels_last", k_coef_lip=2.5, theta_initializer="ones"
        )
        hh.build((32, 32, 16))
        assert hh.theta.shape == (8,)
        np.testing.assert_equal(hh.theta.numpy(), 1)

        # Check serialization
        hh = Householder(theta_initializer="glorot_uniform")
        check_serialization(hh)

        # Instantiation error because of wrong data format
        with self.assertRaisesRegex(RuntimeError, "data format is supported"):
            Householder(data_format="channels_first")

    def test_theta_zero_dense_potentials(self):
        """Householder with theta=0 on 2-D tensor (bs, n).
        Theta=0 means Id if z2 > 0, and reflection if z2 < 0.
        """
        hh = Householder(theta_initializer="zeros")

        bs = np.random.randint(64, 512)
        n = np.random.randint(1, 1024) * 2

        # Case 1: hh(x) = x   (identity case, z2 > 0)
        z1 = keras.random.normal((bs, n // 2))
        z2 = keras.random.uniform((bs, n // 2))
        x = K.concatenate([z1, z2], axis=-1)
        np.testing.assert_allclose(hh(x), x)

        # Case 2: hh(x) = [z1, -z2]   (reflection across z1 axis, z2 < 0)
        z1 = keras.random.normal((bs, n // 2))
        z2 = -keras.random.uniform((bs, n // 2))
        x = K.concatenate([z1, z2], axis=-1)
        expected_output = K.concatenate([z1, -z2], axis=-1)
        np.testing.assert_allclose(hh(x), expected_output)

    def test_theta_pi_dense_potentials(self):
        """Householder with theta=pi on 2-D tensor (bs, n).
        Theta=pi means Id if z1 < 0, and reflection if z1 > 0.
        """
        hh = Householder(theta_initializer=keras.initializers.Constant(np.pi))

        bs = np.random.randint(64, 512)
        n = np.random.randint(1, 1024) * 2

        # Case 1: hh(x) = x   (identity case, z1 < 0)
        z1 = -keras.random.uniform((bs, n // 2))
        z2 = keras.random.normal((bs, n // 2))
        x = K.concatenate([z1, z2], axis=-1)
        np.testing.assert_allclose(hh(x), x, atol=1e-6)

        # Case 2: hh(x) = [z1, -z2]   (reflection across z2 axis, z1 > 0)
        z1 = keras.random.uniform((bs, n // 2))
        z2 = keras.random.normal((bs, n // 2))
        x = K.concatenate([z1, z2], axis=-1)
        expected_output = K.concatenate([-z1, z2], axis=-1)
        np.testing.assert_allclose(hh(x), expected_output, atol=1e-6)

    def test_theta_90_dense_potentials(self):
        """Householder with theta=pi/2 on 2-D tensor (bs, n).
        Theta=pi/2 is equivalent to GroupSort2: Id if z1 < z2, and reflection if z1 > z2
        """
        hh = Householder()

        bs = np.random.randint(64, 512)
        n = np.random.randint(1, 1024) * 2

        # Case 1: hh(x) = x   (identity case, z1 < z2)
        z1 = -keras.random.normal((bs, n // 2))
        z2 = z1 + keras.random.uniform((bs, n // 2))
        x = K.concatenate([z1, z2], axis=-1)
        np.testing.assert_allclose(hh(x), x)

        # Case 2: hh(x) = reflection(x)   (if z1 > z2)
        z1 = keras.random.normal((bs, n // 2))
        z2 = z1 - keras.random.uniform((bs, n // 2))
        x = K.concatenate([z1, z2], axis=-1)
        expected_output = K.concatenate([z2, z1], axis=-1)
        np.testing.assert_allclose(hh(x), expected_output, atol=1e-6)

    def test_theta_zero_conv_potentials(self):
        """Householder with theta=0 on 4-D tensor (bs, h, w, c).
        Theta=0 means Id if z2 > 0, and reflection if z2 < 0.
        """
        hh = Householder(theta_initializer="zeros")

        bs = np.random.randint(32, 128)
        h, w = np.random.randint(1, 64), np.random.randint(1, 64)
        c = np.random.randint(1, 64) * 2

        # Case 1: hh(x) = x   (identity case, z2 > 0)
        z1 = keras.random.normal((bs, h, w, c // 2))
        z2 = keras.random.uniform((bs, h, w, c // 2))
        x = K.concatenate([z1, z2], axis=-1)
        np.testing.assert_allclose(hh(x), x)

        # Case 2: hh(x) = [z1, -z2]   (reflection across z1 axis, z2 < 0)
        z1 = keras.random.normal((bs, h, w, c // 2))
        z2 = -keras.random.uniform((bs, h, w, c // 2))
        x = K.concatenate([z1, z2], axis=-1)
        expected_output = K.concatenate([z1, -z2], axis=-1)
        np.testing.assert_allclose(hh(x), expected_output)

    def test_theta_pi_conv_potentials(self):
        """Householder with theta=pi on 4-D tensor (bs, h, w, c).
        Theta=pi means Id if z1 < 0, and reflection if z1 > 0.
        """
        hh = Householder(theta_initializer=keras.initializers.Constant(np.pi))

        bs = np.random.randint(32, 128)
        h, w = np.random.randint(1, 64), np.random.randint(1, 64)
        c = np.random.randint(1, 64) * 2

        # Case 1: hh(x) = x   (identity case, z1 < 0)
        z1 = -keras.random.uniform((bs, h, w, c // 2))
        z2 = keras.random.normal((bs, h, w, c // 2))
        x = K.concatenate([z1, z2], axis=-1)
        np.testing.assert_allclose(hh(x), x, atol=1e-6)

        # Case 2: hh(x) = [z1, -z2]   (reflection across z2 axis, z1 > 0)
        z1 = keras.random.uniform((bs, h, w, c // 2))
        z2 = keras.random.normal((bs, h, w, c // 2))
        x = K.concatenate([z1, z2], axis=-1)
        expected_output = K.concatenate([-z1, z2], axis=-1)
        np.testing.assert_allclose(hh(x), expected_output, atol=1e-6)

    def test_theta_90_conv_potentials(self):
        """Householder with theta=pi/2 on 4-D tensor (bs, h, w, c).
        Theta=pi/2 is equivalent to GroupSort2: Id if z1 < z2, and reflection if z1 > z2
        """
        hh = Householder()

        bs = np.random.randint(32, 128)
        h, w = np.random.randint(1, 64), np.random.randint(1, 64)
        c = np.random.randint(1, 64) * 2

        # Case 1: hh(x) = x   (identity case, z1 < z2)
        z1 = -keras.random.normal((bs, h, w, c // 2))
        z2 = z1 + keras.random.uniform((bs, h, w, c // 2))
        x = K.concatenate([z1, z2], axis=-1)
        np.testing.assert_allclose(hh(x), x)

        # Case 2: hh(x) = reflection(x)   (if z1 > z2)
        z1 = keras.random.normal((bs, h, w, c // 2))
        z2 = z1 - keras.random.uniform((bs, h, w, c // 2))
        x = K.concatenate([z1, z2], axis=-1)
        expected_output = K.concatenate([z2, z1], axis=-1)
        np.testing.assert_allclose(hh(x), expected_output, atol=1e-6)

    def test_idempotence(self):
        """Assert idempotence of Householder activation: hh(hh(x)) = hh(x)"""
        hh = Householder(theta_initializer="glorot_uniform")

        bs = np.random.randint(32, 128)
        h, w = np.random.randint(1, 64), np.random.randint(1, 64)
        c = np.random.randint(1, 32) * 2
        x = keras.random.normal((bs, h, w, c))

        # Run two times the HH activation and compare both outputs
        y = hh(x)
        z = hh(y)
        np.testing.assert_allclose(y, z)
