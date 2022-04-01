# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest
from addict import Dict as ADDict
from anomalib.deploy import OpenVINOInferencer

from ote_anomalib.openvino import OTEOpenVINOAnomalyDataloader, OpenVINOAnomalyTask
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
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType


class MockOTEOpenVINOAnomalyDataloader(OTEOpenVINOAnomalyDataloader):
    def __init__(self):
        pass


class MockOpenVINOAnomalyTask(OpenVINOAnomalyTask):
    def __init__(self):
        pass


class MockOpenVINOInferencer(OpenVINOInferencer):
    def __init__(self):
        pass


class TestOTEOpenVINOAnomalyDataloaderInputParamsValidation:
    @e2e_pytest_unit
    def test_openvino_anomaly_data_loader_init_params_validation(self):
        """
        <b>Description:</b>
        Check OTEOpenVINOAnomalyDataloader object initialization parameters validation

        <b>Input data:</b>
        OTEOpenVINOAnomalyDataloader object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OTEOpenVINOAnomalyDataloader object initialization parameter
        """
        correct_values_dict = {
            "config": ADDict(),
            "dataset": DatasetEntity,
            "inferencer": MockOpenVINOInferencer(),
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "config" parameter
            ("config", unexpected_str),
            # Unexpected string is specified as "dataset" parameter
            ("dataset", unexpected_str),
            # Unexpected string is specified as "inferencer" parameter
            ("inferencer", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=OTEOpenVINOAnomalyDataloader,
        )

    @e2e_pytest_unit
    def test_openvino_anomaly_data_loader_getitem_params_validation(self):
        """
        <b>Description:</b>
        Check OTEOpenVINOAnomalyDataloader class "__getitem__" method parameters validation

        <b>Input data:</b>
        OTEOpenVINOAnomalyDataloader object, "index" non-integer object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "__getitem__" method input parameter
        """
        data_loader = MockOTEOpenVINOAnomalyDataloader()
        with pytest.raises(ValueError):
            data_loader.__getitem__(index="unexpected string")  # type: ignore


class TestOpenVINOAnomalyTaskInputParamsValidation:
    @staticmethod
    def model() -> ModelEntity:
        dataset_entity = DatasetEntity()
        model_configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(header="model configurable parameters"),
            label_schema=LabelSchemaEntity(),
        )
        return ModelEntity(train_dataset=dataset_entity, configuration=model_configuration)

    @e2e_pytest_unit
    def test_openvino_anomaly_task_init_params_validation(self):
        """
        <b>Description:</b>
        Check OpenVINOAnomalyTask object initialization parameters validation

        <b>Input data:</b>
        "task_environment" non-TaskEnvironment object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OpenVINOAnomalyTask object initialization parameter
        """
        with pytest.raises(ValueError):
            OpenVINOAnomalyTask(task_environment="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_openvino_anomaly_task_infer_params_validation(self):
        """
        <b>Description:</b>
        Check OpenVINOAnomalyTask class "infer" method parameters validation

        <b>Input data:</b>
        OpenVINOAnomalyTask object, "infer" method input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "infer" method input parameter
        """
        openvino_task = MockOpenVINOAnomalyTask()
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
            class_or_function=openvino_task.infer,
        )

    @e2e_pytest_unit
    def test_openvino_anomaly_task_evaluate_params_validation(self):
        """
        <b>Description:</b>
        Check OpenVINOAnomalyTask class "evaluate" method parameters validation

        <b>Input data:</b>
        OpenVINOAnomalyTask object, "evaluate" method input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "evaluate" method input parameter
        """
        model = self.model()
        output_result_set = ResultSetEntity(
            model=model, ground_truth_dataset=model.train_dataset, prediction_dataset=model.train_dataset
        )
        openvino_task = MockOpenVINOAnomalyTask()
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
            class_or_function=openvino_task.evaluate,
        )

    @e2e_pytest_unit
    def test_openvino_anomaly_task_optimize_params_validation(self):
        """
        <b>Description:</b>
        Check OpenVINOAnomalyTask class "optimize" method parameters validation

        <b>Input data:</b>
        OpenVINOAnomalyTask object, "optimize" method input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "optimize" method input parameter
        """
        openvino_task = MockOpenVINOAnomalyTask()
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
            class_or_function=openvino_task.optimize,
        )

    @e2e_pytest_unit
    def test_openvino_anomaly_task_deploy_params_validation(self):
        """
        <b>Description:</b>
        Check OpenVINOAnomalyTask class "deploy" method parameters validation

        <b>Input data:</b>
        OpenVINOAnomalyTask object, "output_model" non-ModelEntity object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "deploy" method input parameter
        """
        openvino_task = MockOpenVINOAnomalyTask()
        with pytest.raises(ValueError):
            openvino_task.deploy(output_model="unexpected string")  # type: ignore
