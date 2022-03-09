# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment


class LightningSpeechToTextLogger(LightningLoggerBase):
    """Simple logger for pytorch lightning pipeline."""
    def __init__(self, monitor, mode="min"):
        super().__init__()
        assert mode in ["min", "max"]
        self.monitor_fn = min if mode == "min" else max
        self.monitor = monitor
        self.reset()

    @property
    def name(self):
        return "STTLogger"

    @rank_zero_only
    def log_hyperparams(self, params):
        self.params = params

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        if not self.monitor in metrics:
            return
        self.step = step if step is not None else self.step + 1
        self.log[self.step] = metrics
        if self.best_metric is None:
            self.best_metric = metrics[self.monitor]
        self.best_metric = self.monitor_fn(self.best_metric, metrics[self.monitor])

    def reset(self):
        self.log = {}
        self.best_metric = None
        self.params = None
        self.step = 0

    @property
    @rank_zero_experiment
    def experiment(self):
        pass

    @property
    def version(self):
        return "0.1"
