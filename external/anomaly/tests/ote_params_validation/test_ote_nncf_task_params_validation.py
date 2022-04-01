# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest

from ote_anomalib.nncf_task import AnomalyNNCFTask
from ote_sdk.configuration.configurable_parameters import ConfigurableParameters
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import ModelConfiguration, ModelEntity
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType


class MockAnomalyNNCFTask(AnomalyNNCFTask):
    def __init__(self):
        pass


class TestAnomalyNNCFTaskInputParamsValidation:
    @staticmethod
    def model() -> ModelEntity:
        dataset_entity = DatasetEntity()
        model_configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(header="model configurable parameters"),
            label_schema=LabelSchemaEntity(),
        )
        return ModelEntity(train_dataset=dataset_entity, configuration=model_configuration)

    @e2e_pytest_unit
    def test_anomaly_nncf_task_init_params_validation(self):
        """
        <b>Description:</b>
        Check AnomalyNNCFTask object initialization parameters validation

        <b>Input data:</b>
        "task_environment" non-TaskEnvironment object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        AnomalyNNCFTask object initialization parameter
        """
        with pytest.raises(ValueError):
            AnomalyNNCFTask(task_environment="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_anomaly_nncf_task_load_model_params_validation(self):
        """
        <b>Description:</b>
        Check AnomalyNNCFTask class "load_model" method parameters validation

        <b>Input data:</b>
        AnomalyNNCFTask object, "ote_model" non-ModelEntity object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "load_model" method input parameter
        """
        nncf_task = MockAnomalyNNCFTask()
        with pytest.raises(ValueError):
            nncf_task.load_model(ote_model="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_anomaly_nncf_task_optimize_params_validation(self):
        """
        <b>Description:</b>
        Check AnomalyNNCFTask class "optimize" method parameters validation

        <b>Input data:</b>
        AnomalyNNCFTask object, "optimize" method input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "optimize" method input parameter
        """
        nncf_task = MockAnomalyNNCFTask()
        correct_values_dict = {
            "optimization_type": OptimizationType.NNCF,
            "dataset": DatasetEntity(),
            "output_model": self.model(),
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "optimization_type" parameter
            ("optimization_type", unexpected_str),
            # Unexpected string is specified as "dataset" parameter
            ("dataset", unexpected_str),
            # Unexpected string is specified as "output_model" parameter
            ("output_model", unexpected_str),
            # Unexpected string is specified as "optimization_parameters" parameter
            ("optimization_parameters", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=nncf_task.optimize,
        )
