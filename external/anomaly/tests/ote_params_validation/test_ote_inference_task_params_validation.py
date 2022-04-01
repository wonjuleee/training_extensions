# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest

from ote_anomalib.inference_task import AnomalyInferenceTask
from ote_sdk.configuration.configurable_parameters import ConfigurableParameters
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import ModelConfiguration, ModelEntity
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType


class MockAnomalyInferenceTask(AnomalyInferenceTask):
    def __init__(self):
        pass


class TestAnomalyInferenceTaskInputParamsValidation:
    @staticmethod
    def model() -> ModelEntity:
        dataset_entity = DatasetEntity()
        model_configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(header="model configurable parameters"),
            label_schema=LabelSchemaEntity(),
        )
        return ModelEntity(train_dataset=dataset_entity, configuration=model_configuration)

    @e2e_pytest_unit
    def test_anomaly_inference_task_init_params_validation(self):
        """
        <b>Description:</b>
        Check AnomalyInferenceTask object initialization parameters validation

        <b>Input data:</b>
        "task_environment" non-TaskEnvironment object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        AnomalyInferenceTask object initialization parameter
        """
        with pytest.raises(ValueError):
            AnomalyInferenceTask(task_environment="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_anomaly_inference_task_load_model_params_validation(self):
        """
        <b>Description:</b>
        Check AnomalyInferenceTask class "load_model" method parameters validation

        <b>Input data:</b>
        AnomalyInferenceTask object, "ote_model" non-ModelEntity object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "load_model" method input parameter
        """
        inference_task = MockAnomalyInferenceTask()
        with pytest.raises(ValueError):
            inference_task.load_model(ote_model="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_anomaly_inference_task_infer_params_validation(self):
        """
        <b>Description:</b>
        Check AnomalyInferenceTask class "infer" method parameters validation

        <b>Input data:</b>
        AnomalyInferenceTask object, "infer" method input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "infer" method input parameter
        """
        inference_task = MockAnomalyInferenceTask()
        correct_values_dict = {
            "dataset": DatasetEntity(),
            "inference_parameters": InferenceParameters(),
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "dataset" parameter
            ("dataset", unexpected_str),
            # Unexpected string is specified as "inference_parameters" parameter
            ("inference_parameters", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=inference_task.infer,
        )

    @e2e_pytest_unit
    def test_anomaly_inference_task_evaluate_params_validation(self):
        """
        <b>Description:</b>
        Check AnomalyInferenceTask class "evaluate" method parameters validation

        <b>Input data:</b>
        AnomalyInferenceTask object, "evaluate" method input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "evaluate" method input parameter
        """
        model = self.model()
        output_result_set = ResultSetEntity(
            model=model, ground_truth_dataset=model.train_dataset, prediction_dataset=model.train_dataset
        )
        inference_task = MockAnomalyInferenceTask()
        correct_values_dict = {"output_resultset": output_result_set}
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "output_resultset" parameter
            ("output_resultset", unexpected_int),
            # Unexpected integer is specified as "evaluation_metric" parameter
            ("evaluation_metric", unexpected_int),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=inference_task.evaluate,
        )

    @e2e_pytest_unit
    def test_anomaly_inference_task_export_params_validation(self):
        """
        <b>Description:</b>
        Check AnomalyInferenceTask class "export" method parameters validation

        <b>Input data:</b>
        AnomalyInferenceTask object, "export" method input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "export" method input parameter
        """
        inference_task = MockAnomalyInferenceTask()
        correct_values_dict = {"export_type": ExportType.OPENVINO, "output_model": self.model()}
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "export_type" parameter
            ("export_type", unexpected_int),
            # Unexpected integer is specified as "output_model" parameter
            ("output_model", unexpected_int),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=inference_task.export,
        )

    @e2e_pytest_unit
    def test_anomaly_inference_task_save_model_params_validation(self):
        """
        <b>Description:</b>
        Check AnomalyInferenceTask class "save_model" method parameters validation

        <b>Input data:</b>
        AnomalyInferenceTask object, "output_model" non-ModelEntity object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "save_model" method input parameter
        """
        inference_task = MockAnomalyInferenceTask()
        with pytest.raises(ValueError):
            inference_task.save_model(output_model="unexpected string")  # type: ignore
