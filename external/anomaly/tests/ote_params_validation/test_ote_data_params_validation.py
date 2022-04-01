# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest
from omegaconf import DictConfig

from ote_anomalib.data.data import OTEAnomalyDataset, OTEAnomalyDataModule
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.model_template import TaskType
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


class MockAnomalyDataset(OTEAnomalyDataset):
    def __init__(self):
        pass


class TestOTEAnomalyDatasetInputParamsValidation:
    @e2e_pytest_unit
    def test_ote_anomaly_dataset_init_params_validation(self):
        """
        <b>Description:</b>
        Check OTEAnomalyDataset object initialization parameters validation

        <b>Input data:</b>
        OTEAnomalyDataset object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OTEAnomalyDataset object initialization parameter
        """
        correct_values_dict = {
            "config": DictConfig({"some": "data"}),
            "dataset": DatasetEntity(),
            "task_type": TaskType.ANOMALY_SEGMENTATION,
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "config" parameter
            ("config", unexpected_str),
            # Unexpected string is specified as "dataset" parameter
            ("dataset", unexpected_str),
            # Unexpected string is specified as "task_type" parameter
            ("task_type", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=OTEAnomalyDataset,
        )

    @e2e_pytest_unit
    def test_ote_anomaly_dataset_get_item_params_validation(self):
        """
        <b>Description:</b>
        Check OTEAnomalyDataset class "__getitem__" method parameters validation

        <b>Input data:</b>
        OTEAnomalyDataset object, "index" non-integer object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "__getitem__" method input parameter
        """
        ote_anomaly_dataset = MockAnomalyDataset()
        with pytest.raises(ValueError):
            ote_anomaly_dataset.__getitem__("unexpected string")  # type: ignore


class TestOTEAnomalyDataModuleInputParamsValidation:
    @e2e_pytest_unit
    def test_ote_anomaly_data_module_init_params_validation(self):
        """
        <b>Description:</b>
        Check OTEAnomalyDataModule object initialization parameters validation

        <b>Input data:</b>
        OTEAnomalyDataModule object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OTEAnomalyDataModule object initialization parameter
        """
        correct_values_dict = {
            "config": DictConfig({"some": "data"}),
            "dataset": DatasetEntity(),
            "task_type": TaskType.ANOMALY_SEGMENTATION,
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "config" parameter
            ("config", unexpected_str),
            # Unexpected string is specified as "dataset" parameter
            ("dataset", unexpected_str),
            # Unexpected string is specified as "task_type" parameter
            ("task_type", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=OTEAnomalyDataModule,
        )
