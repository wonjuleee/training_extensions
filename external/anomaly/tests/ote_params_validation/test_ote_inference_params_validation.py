# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytorch_lightning as pl
from anomalib.models import AnomalyModule

from ote_anomalib.callbacks.inference import AnomalyInferenceCallback
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.model_template import TaskType
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


class MockAnomalyCallback(AnomalyInferenceCallback):
    def __init__(self):
        pass


class MockAnomalyModule(AnomalyModule):
    def __init__(self):
        pass


class TestAnomalyInferenceCallbackInputParamsValidation:
    @e2e_pytest_unit
    def test_anomaly_inference_callback_init_params_validation(self):
        """
        <b>Description:</b>
        Check AnomalyInferenceCallback object initialization parameters validation

        <b>Input data:</b>
        AnomalyInferenceCallback object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        AnomalyInferenceCallback object initialization parameter
        """
        label = LabelEntity(name="test label", domain=Domain.ANOMALY_SEGMENTATION)
        correct_values_dict = {
            "ote_dataset": DatasetEntity(),
            "labels": [label],
            "task_type": TaskType.ANOMALY_SEGMENTATION,
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "ote_dataset" parameter
            ("ote_dataset", unexpected_str),
            # Unexpected string is specified as "labels" parameter
            ("labels", unexpected_str),
            # Unexpected string is specified as nested label
            ("labels", [label, unexpected_str]),
            # Unexpected string is specified as "task_type" parameter
            ("task_type", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=AnomalyInferenceCallback,
        )

    @e2e_pytest_unit
    def test_anomaly_inference_callback_on_predict_epoch_end_params_validation(self):
        """
        <b>Description:</b>
        Check AnomalyInferenceCallback class "on_predict_epoch_end" method parameters validation

        <b>Input data:</b>
        AnomalyInferenceCallback object, "on_predict_epoch_end" method input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "on_predict_epoch_end" method input parameter
        """
        anomaly_callback = MockAnomalyCallback()
        correct_values_dict = {
            "_trainer": pl.Trainer(),
            "pl_module": MockAnomalyModule(),
            "outputs": ["some", "outputs"],
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "_trainer" parameter
            ("_trainer", unexpected_str),
            # Unexpected string is specified as "pl_module" parameter
            ("pl_module", unexpected_str),
            # Unexpected string is specified as "outputs" parameter
            ("outputs", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=anomaly_callback.on_predict_epoch_end,
        )
