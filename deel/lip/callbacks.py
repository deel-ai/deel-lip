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
import numpy as np
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
            if isinstance(layer, Condensable) or hasattr(layer, "condense"):
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
        target: str = "kernel",
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
            target: describe what to monitor, can either "kernel" or "wbar". Setting
                to "kernel" check values of the unconstrained weights while setting to
                "wbar" check values of the constrained weights (allowing to check if
                the parameters are correct to ensure lipschitz constraint)
            what: either "max", which display the largest singular value over the
                training process, or "all", which plot the distribution of all singular
                values.
            on_epoch: if True apply the constraint between epochs.
            on_batch: if True apply constraints between batches.
        """
        self.on_epoch = on_epoch
        self.on_batch = on_batch
        assert target in {"kernel", "wbar"}
        self.target = target
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
            if (
                (self.target == "kernel")
                and (self.what == "max")
                and hasattr(layer, "sig")
            ):
                sig = layer.sig[0, 0]
            elif hasattr(layer, self.target):
                kernel = getattr(layer, self.target)
                w_shape = kernel.shape.as_list()
                sigmas = tf.linalg.svd(
                    tf.keras.backend.reshape(kernel, [-1, w_shape[-1]]),
                    full_matrices=False,
                    compute_uv=False,
                ).numpy()
                sig = sigmas[0]
            else:
                raise RuntimeWarning(
                    f"[MonitorCallback] layer {layer_name} has no "
                    f"attribute {self.target}"
                )
                return
            if self.what == "max":
                with self.file_writer.as_default():
                    result = tf.summary.scalar(
                        f"{layer_name}_{self.target}_sigmas", sig, step=step
                    )
            else:
                with self.file_writer.as_default():
                    result = tf.summary.histogram(
                        f"{layer_name}_{self.target}_sigmas",
                        sigmas,
                        step=step,
                        buckets=None,
                        description=(
                            f"distribution of singular values for layer "
                            f"{layer_name}"
                        ),
                    )
            if not result:
                raise RuntimeWarning(
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


class LossParamScheduler(Callback):
    def __init__(self, param_name, fp, xp, step=0):
        """
        Scheduler to modify a loss parameter during training. It uses a linear
        interpolation (defined by fp and xp) depending on the optimization step.

        Args:
            param_name (str): name of the parameter of the loss to tune. Must be a
                tf.Variable.
            fp (list): values of the loss parameter as steps given by the xp.
            xp (list): step where the parameter equals fp.
            step (int): step value, for serialization/deserialization purposes.
        """
        self.xp = xp
        self.fp = fp
        self.step = step
        self.param_name = param_name

    def on_train_batch_begin(self, batch: int, logs=None):
        new_value = np.interp(self.step, self.xp, self.fp)
        self.model.loss.__getattribute__(self.param_name).assign(new_value)
        self.step += 1
        super(LossParamScheduler, self).on_train_batch_end(batch, logs)

    def get_config(self):
        return {
            "xp": self.xp,
            "fp": self.fp,
            "step": self.step,
            "param_name": self.param_name,
        }


class LossParamLog(Callback):
    def __init__(self, param_name, rate=1):
        """
        Logger to print values of a loss parameter at each epoch.

        Args:
            param_name (str): name of the parameter of the loss to log.
            rate (int): logging rate (in epochs)
        """
        self.param_name = param_name
        self.rate = rate

    def on_epoch_end(self, epoch: int, logs=None):
        if epoch % self.rate == 0:
            tf.print(
                "\n",
                self.model.loss.name,
                self.param_name,
                self.model.loss.__getattribute__(self.param_name),
            )
        super(LossParamLog, self).on_train_batch_end(epoch, logs)

    def get_config(self):
        return {
            "param_name": self.param_name,
            "rate": self.rate,
        }
