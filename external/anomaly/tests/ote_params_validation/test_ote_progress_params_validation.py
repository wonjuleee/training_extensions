# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import pytorch_lightning as pl
from anomalib.models import AnomalyModule

from ote_anomalib.callbacks.progress import ProgressCallback
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


class MockAnomalyModule(AnomalyModule):
    def __init__(self):
        pass


class TestProgressCallbackInputParamsValidation:
    @e2e_pytest_unit
    def test_progress_callback_init_params_validation(self):
        """
        <b>Description:</b>
        Check ProgressCallback object initialization parameters validation

        <b>Input data:</b>
        "parameters" non-Union[TrainParameters, InferenceParameters] object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        ProgressCallback object initialization parameter
        """
        with pytest.raises(ValueError):
            ProgressCallback(parameters="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_progress_callback_on_train_start_params_validation(self):
        """
        <b>Description:</b>
        Check ProgressCallback class "on_train_start" method parameters validation

        <b>Input data:</b>
        ProgressCallback object, "on_train_start" method input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "on_train_start" method input parameter
        """
        progress_callback = ProgressCallback()
        correct_values_dict = {
            "trainer": pl.Trainer(),
            "pl_module": MockAnomalyModule(),
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "_trainer" parameter
            ("trainer", unexpected_str),
            # Unexpected string is specified as "pl_module" parameter
            ("pl_module", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=progress_callback.on_train_start,
        )

    @e2e_pytest_unit
    def test_progress_callback_on_predict_start_params_validation(self):
        """
        <b>Description:</b>
        Check ProgressCallback class "on_predict_start" method parameters validation

        <b>Input data:</b>
        ProgressCallback object, "on_predict_start" method input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "on_predict_start" method input parameter
        """
        progress_callback = ProgressCallback()
        correct_values_dict = {
            "trainer": pl.Trainer(),
            "pl_module": MockAnomalyModule(),
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "_trainer" parameter
            ("trainer", unexpected_str),
            # Unexpected string is specified as "pl_module" parameter
            ("pl_module", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=progress_callback.on_predict_start,
        )

    @e2e_pytest_unit
    def test_progress_callback_on_test_start_params_validation(self):
        """
        <b>Description:</b>
        Check ProgressCallback class "on_test_start" method parameters validation

        <b>Input data:</b>
        ProgressCallback object, "on_test_start" method input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "on_test_start" method input parameter
        """
        progress_callback = ProgressCallback()
        correct_values_dict = {
            "trainer": pl.Trainer(),
            "pl_module": MockAnomalyModule(),
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "_trainer" parameter
            ("trainer", unexpected_str),
            # Unexpected string is specified as "pl_module" parameter
            ("pl_module", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=progress_callback.on_test_start,
        )

    @e2e_pytest_unit
    def test_progress_callback_on_train_batch_end_params_validation(self):
        """
        <b>Description:</b>
        Check ProgressCallback class "on_train_batch_end" method parameters validation

        <b>Input data:</b>
        ProgressCallback object, "on_train_batch_end" method input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "on_train_batch_end" method input parameter
        """
        progress_callback = ProgressCallback()
        correct_values_dict = {
            "trainer": pl.Trainer(),
            "pl_module": MockAnomalyModule(),
            "outputs": ["some", "outputs"],
            "batch": "batch data",
            "batch_idx": 1,
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "_trainer" parameter
            ("trainer", unexpected_str),
            # Unexpected string is specified as "pl_module" parameter
            ("pl_module", unexpected_str),
            # Unexpected string is specified as "outputs" parameter
            ("outputs", unexpected_str),
            # Unexpected string is specified as "batch_idx" parameter
            ("batch_idx", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=progress_callback.on_train_batch_end,
        )

    @e2e_pytest_unit
    def test_progress_callback_on_predict_batch_end_params_validation(self):
        """
        <b>Description:</b>
        Check ProgressCallback class "on_predict_batch_end" method parameters validation

        <b>Input data:</b>
        ProgressCallback object, "on_predict_batch_end" method input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "on_predict_batch_end" method input parameter
        """
        progress_callback = ProgressCallback()
        correct_values_dict = {
            "trainer": pl.Trainer(),
            "pl_module": MockAnomalyModule(),
            "outputs": ["some", "outputs"],
            "batch": "batch data",
            "batch_idx": 1,
            "dataloader_idx": 1,
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "_trainer" parameter
            ("trainer", unexpected_str),
            # Unexpected string is specified as "pl_module" parameter
            ("pl_module", unexpected_str),
            # Unexpected string is specified as "outputs" parameter
            ("outputs", unexpected_str),
            # Unexpected string is specified as "batch_idx" parameter
            ("batch_idx", unexpected_str),
            # Unexpected string is specified as "dataloader_idx" parameter
            ("dataloader_idx", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=progress_callback.on_predict_batch_end,
        )

    @e2e_pytest_unit
    def test_progress_callback_on_test_batch_end_params_validation(self):
        """
        <b>Description:</b>
        Check ProgressCallback class "on_test_batch_end" method parameters validation

        <b>Input data:</b>
        ProgressCallback object, "on_test_batch_end" method input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "on_test_batch_end" method input parameter
        """
        progress_callback = ProgressCallback()
        correct_values_dict = {
            "trainer": pl.Trainer(),
            "pl_module": MockAnomalyModule(),
            "outputs": ["some", "outputs"],
            "batch": "batch data",
            "batch_idx": 1,
            "dataloader_idx": 1,
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "_trainer" parameter
            ("trainer", unexpected_str),
            # Unexpected string is specified as "pl_module" parameter
            ("pl_module", unexpected_str),
            # Unexpected string is specified as "outputs" parameter
            ("outputs", unexpected_str),
            # Unexpected string is specified as "batch_idx" parameter
            ("batch_idx", unexpected_str),
            # Unexpected string is specified as "dataloader_idx" parameter
            ("dataloader_idx", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=progress_callback.on_test_batch_end,
        )
