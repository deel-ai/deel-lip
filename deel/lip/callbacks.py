# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module contains callbacks that can be added to keras training process.
"""
import os
from typing import Optional, Dict, Iterable

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from .layers import Condensable


class CondenseCallback(Callback):
    def __init__(self, on_epoch: bool = True, on_batch: bool = False):
        """
        Automatically condense layers of a model on batches/epochs. Condensing a layer
        consists in overwriting the kernel with the constrained weights. This prevents
        the explosion/vanishing of values inside the original kernel.

        Warning:
            Overwriting the kernel may disturb the optimizer, especially if it has a
            non-zero momentum.

        Args:
            on_epoch: if True apply the constraint between epochs
            on_batch: if True apply constraints between batches
        """
        super().__init__()
        self.on_epoch = on_epoch
        self.on_batch = on_batch

    def _condense_model(self):
        for layer in self.model.layers:
            if isinstance(layer, Condensable) or hasattr(layer, 'condense'):
                layer.condense()

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, float]] = None):
        if self.on_batch:
            self._condense_model()
        super(CondenseCallback, self).on_train_batch_end(batch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        if self.on_epoch:
            self._condense_model()
        super(CondenseCallback, self).on_epoch_end(epoch, logs)

    def get_config(self):
        config = {"on_epoch": self.on_epoch, "on_batch": self.on_batch}
        base_config = super(CondenseCallback, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MonitorCallback(Callback):
    def __init__(
        self,
        monitored_layers: Iterable[str],
        logdir: str,
        what: str = "max",
        on_epoch: bool = True,
        on_batch: bool = False,
    ):
        """
        Allow to monitor the singular values of specified layers during training. This
        analyze the singular values of the original kernel (before reparametrization).
        Two modes can be chosen: "max" plots the largest singular value over training,
        while "all" plots the distribution of the singular values over training (series
        of distribution).

        Args:
            monitored_layers: list of layer name to monitor.
            logdir: path to the logging directory.
            what: either "max", which display the largest singular value over the
                training process, or "all", which plot the distribution of all singular
                values.
            on_epoch: if True apply the constraint between epochs.
            on_batch: if True apply constraints between batches.
        """
        self.on_epoch = on_epoch
        self.on_batch = on_batch
        assert what in {"max", "all"}
        self.what = what
        self.logdir = logdir
        self.file_writer = tf.summary.create_file_writer(
            os.path.join(logdir, "metrics")
        )
        self.monitored_layers = monitored_layers
        if on_batch and on_epoch:
            self.on_epoch = False  # avoid display bug (inconsistent steps)
        self.epochs = 0
        super().__init__()

    def _monitor(self, step):
        step = self.params["steps"] * self.epochs + step
        for layer_name in self.monitored_layers:
            layer = self.model.get_layer(layer_name)
            if (self.what == "max") and hasattr(layer, "sig"):
                sig = layer.sig[0, 0]
            elif hasattr(layer, "kernel"):
                kernel = layer.kernel
                w_shape = kernel.shape.as_list()
                sigmas = tf.linalg.svd(
                    tf.keras.backend.reshape(kernel, [-1, w_shape[-1]]),
                    full_matrices=False,
                    compute_uv=False,
                ).numpy()
                sig = sigmas[0]
            else:
                RuntimeWarning("[MonitorCallback] unsupported layer")
                return
            if self.what == "max":
                with self.file_writer.as_default():
                    result = tf.summary.scalar("%s_sigma" % layer_name, sig, step=step)
            else:
                with self.file_writer.as_default():
                    result = tf.summary.histogram(
                        "%s_sigmas" % layer_name,
                        sigmas,
                        step=step,
                        buckets=None,
                        description="distribution of singular values for layer %s"
                        % layer_name,
                    )
            if not result:
                RuntimeWarning(
                    "[MonitorCallback] unable to find filewriter, no logs were written,"
                )

    def on_train_batch_end(self, batch, logs=None):
        if self.on_batch:
            self._monitor(batch)
        super(MonitorCallback, self).on_train_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self.epochs += 1
        if self.on_epoch:
            self._monitor(epoch)
        super(MonitorCallback, self).on_epoch_end(epoch, logs)

    def get_config(self):
        config = {
            "on_epoch": self.on_epoch,
            "on_batch": self.on_batch,
            "monitored_layers": self.monitored_layers,
        }
        base_config = super(MonitorCallback, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
