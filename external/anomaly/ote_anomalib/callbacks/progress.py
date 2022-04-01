"""
Progressbar Callback for OTE task
"""

# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from ote_sdk.entities.train_parameters import UpdateProgressCallback
from pytorch_lightning.callbacks.progress import ProgressBar


class ProgressCallback(ProgressBar):
    """
    Modifies progress callback to show completion of the entire training step
    """

    def __init__(self, update_progress_callback: UpdateProgressCallback, loading_stage_progress_percentage: int = 0,
                 initialization_stage_progress_percentage: int = 0) -> None:
        super().__init__()
        self.current_epoch: int = 0
        self.max_epochs: int = 0

        if initialization_stage_progress_percentage + loading_stage_progress_percentage >= 100:
            raise RuntimeError('Total optimization progress percentage is more than 100%')
        self.main_stage_percentage = 100 - initialization_stage_progress_percentage - loading_stage_progress_percentage
        self.loading_stage_progress_percentage = loading_stage_progress_percentage
        self.initialization_stage_progress_percentage = initialization_stage_progress_percentage

        self.model_is_loaded = False
        self.model_is_initialized = False
        self.train_ended = False

        self.update_progress_callback = update_progress_callback

    def on_model_loaded(self, stage="train"):
        self.model_is_loaded = True
        self._update_progress(stage)

    def on_model_initialized(self, stage="train"):
        self.model_is_initialized = True
        self._update_progress(stage)

    def on_train_start(self, trainer, pl_module):
        """
        Store max epochs and current epoch from trainer
        """
        super().on_train_start(trainer, pl_module)
        self.on_model_loaded("train")
        self.on_model_initialized("train")

        self.current_epoch = trainer.current_epoch
        self.max_epochs = trainer.max_epochs
        self._update_progress(stage="train")

    def on_predict_start(self, trainer, pl_module):
        """
        Reset progress bar when prediction starts.
        """
        super().on_predict_start(trainer, pl_module)
        self.on_model_loaded("predict")
        self.on_model_initialized("predict")

    def on_test_start(self, trainer, pl_module):
        """
        Reset progress bar when testing starts.
        """
        super().on_test_start(trainer, pl_module)
        self.on_model_loaded("test")
        self.on_model_initialized("test")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Adds training completion percentage to the progress bar
        """
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.current_epoch = trainer.current_epoch
        self._update_progress(stage="train")

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """
        Adds prediction completion percentage to the progress bar
        """
        super().on_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self._update_progress(stage="predict")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """
        Adds testing completion percentage to the progress bar
        """
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self._update_progress(stage="test")

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        self.train_ended = True
        self._update_progress("train")

    def _get_progress(self, stage: str = "train") -> float:
        """
        Get progress for train and test stages.

        Args:
            stage (str, optional): Train or Test stages. Defaults to "train".
        """

        if stage == "train":
            # Progress is calculated on the upper bound (max epoch).
            # Early stopping might stop the training before the progress reaches 100%
            main_stage_relative_progress = 1.0 if self.train_ended else (
                (self.train_batch_idx + self.current_epoch * self.total_train_batches)
                / (self.total_train_batches * self.max_epochs)
            )
        elif stage == "predict":
            main_stage_relative_progress = self.predict_batch_idx / (self.total_predict_batches + 1e-10)
        elif stage == "test":
            main_stage_relative_progress = self.test_batch_idx / (self.total_test_batches + 1e-10)
        else:
            raise ValueError(f"Unknown stage {stage}. Available: train, predict and test")

        progress = main_stage_relative_progress * self.main_stage_percentage
        if self.model_is_loaded:
            progress += self.loading_stage_progress_percentage
        if self.model_is_initialized:
            progress += self.initialization_stage_progress_percentage

        return progress

    def _update_progress(self, stage: str):
        progress = self._get_progress(stage)
        self.update_progress_callback(progress)
