# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from ote_anomalib.train_task import AnomalyTrainingTask
from ote_sdk.configuration.configurable_parameters import ConfigurableParameters
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import ModelConfiguration, ModelEntity
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


class MockAnomalyTrainingTask(AnomalyTrainingTask):
    def __init__(self):
        pass


class TestAnomalyTrainingTaskInputParamsValidation:
    @staticmethod
    def model() -> ModelEntity:
        dataset_entity = DatasetEntity()
        model_configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(header="model configurable parameters"),
            label_schema=LabelSchemaEntity(),
        )
        return ModelEntity(train_dataset=dataset_entity, configuration=model_configuration)

    @e2e_pytest_unit
    def test_anomaly_train_task_train_params_validation(self):
        """
        <b>Description:</b>
        Check AnomalyTrainingTask class "train" method parameters validation

        <b>Input data:</b>
        AnomalyTrainingTask object, "train" method input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "train" method input parameter
        """
        training_task = MockAnomalyTrainingTask()
        correct_values_dict = {
            "dataset": DatasetEntity(),
            "output_model": self.model(),
            "train_parameters": TrainParameters(),
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "dataset" parameter
            ("dataset", unexpected_str),
            # Unexpected string is specified as "output_model" parameter
            ("output_model", unexpected_str),
            # Unexpected string is specified as "train_parameters" parameter
            ("train_parameters", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=training_task.train,
        )
