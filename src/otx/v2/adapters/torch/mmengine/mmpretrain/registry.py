"""OTX adapters.torch.mmengine.mmpretrain.Registry module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mmpretrain.registry import (
    BATCH_AUGMENTS,
    DATA_SAMPLERS,
    DATASETS,
    EVALUATORS,
    HOOKS,
    LOG_PROCESSORS,
    LOOPS,
    METRICS,
    MODEL_WRAPPERS,
    MODELS,
    OPTIM_WRAPPER_CONSTRUCTORS,
    OPTIM_WRAPPERS,
    OPTIMIZERS,
    PARAM_SCHEDULERS,
    RUNNER_CONSTRUCTORS,
    RUNNERS,
    TASK_UTILS,
    TRANSFORMS,
    VISBACKENDS,
    VISUALIZERS,
    WEIGHT_INITIALIZERS,
)

from otx.v2.adapters.torch.mmengine.registry import MMEngineRegistry

REGISTRY_LIST = [
    RUNNERS,
    RUNNER_CONSTRUCTORS,
    LOOPS,
    HOOKS,
    LOG_PROCESSORS,
    OPTIMIZERS,
    OPTIM_WRAPPERS,
    OPTIM_WRAPPER_CONSTRUCTORS,
    PARAM_SCHEDULERS,
    DATASETS,
    DATA_SAMPLERS,
    TRANSFORMS,
    MODELS,
    MODEL_WRAPPERS,
    WEIGHT_INITIALIZERS,
    BATCH_AUGMENTS,
    TASK_UTILS,
    METRICS,
    EVALUATORS,
    VISUALIZERS,
    VISBACKENDS,
]


class MMPretrainRegistry(MMEngineRegistry):
    """Registry for MMPretrain models and related components."""

    def __init__(self, name: str = "mmpretrain") -> None:
        """Initialize a new instance of the `Registry` class.

        Args:
            name (str): The name of the registry. Defaults to "mmpretrain".
        """
        super().__init__(name)
        self._registry_dict = {registry.name: registry for registry in REGISTRY_LIST}