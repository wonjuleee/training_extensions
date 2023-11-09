"""Model builder & get list of available model API for OTX lightning adapter."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fnmatch
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from otx.v2.api.utils.importing import get_files_dict, get_otx_root_path

from .modules.models import MODELS

MODEL_CONFIG_PATH = Path(get_otx_root_path()) / "v2/configs/lightning/models"
MODEL_CONFIGS = get_files_dict(MODEL_CONFIG_PATH)


def get_model(
    model: dict | (DictConfig | str) | None = None,
    checkpoint: str | None = None,
    **kwargs,
) -> torch.nn.Module:
    """Return a torch.nn.Module object based on the provided model configuration or Lightning model api.

    Args:
        model (dict | (DictConfig | str) | None, optional): The model configuration. Can be a dictionary,
            a DictConfig object, or a path to a YAML file containing the configuration.
        checkpoint (str | None, optional): The path to a checkpoint file to load weights from.
        **kwargs (Any): Additional keyword arguments to pass to the `get_model` function.

    Returns:
        torch.nn.Module: The model object.

    """
    kwargs = kwargs or {}
    if isinstance(model, str):
        if model in MODEL_CONFIGS:
            model = MODEL_CONFIGS[model]
        if Path(model).is_file():
            model = OmegaConf.load(model)
    elif isinstance(model, (dict, DictConfig)):
        if not model.get("model", False):
            model = DictConfig(content={"model": model})
        if isinstance(model, dict):
            model = OmegaConf.create(model)
        if getattr(model.model, "name", None) in MODEL_CONFIGS:
            model = MODEL_CONFIGS[model.model["name"]]
            model = OmegaConf.load(model)

    state_dict = None
    if checkpoint is not None:
        model["checkpoint"] = checkpoint
        state_dict = torch.load(checkpoint)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    model_class = MODELS.get(model.model.name)
    if model_class is None:
        msg = f"Current selected model {model.model.name} is not implemented."
        raise NotImplementedError(
            msg,
        )
    return model_class(config=model, state_dict=state_dict)


def list_models(pattern: str | None = None) -> list[str]:
    """Return a list of available model names.

    Args:
        pattern (str | None, optional): A pattern to filter the model names. Defaults to None.

    Returns:
        list[str]: A sorted list of available model names.
    """
    model_list = list(MODEL_CONFIGS.keys())

    if pattern is not None:
        # Always match keys with any postfix.
        model_list = list(set(fnmatch.filter(model_list, pattern + "*")))

    return sorted(model_list)