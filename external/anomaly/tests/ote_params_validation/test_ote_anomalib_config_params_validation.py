# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from omegaconf import DictConfig

from ote_anomalib.configs.anomalib_config import get_anomalib_config, update_anomalib_config
from ote_sdk.configuration.configurable_parameters import ConfigurableParameters
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


class TestAnomalibConfigInputParamsValidation:
    @e2e_pytest_unit
    def test_get_anomalib_config_params_validation(self):
        """
        <b>Description:</b>
        Check "get_anomalib_config" function parameters validation

        <b>Input data:</b>
        "get_anomalib_config" function input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "get_anomalib_config" function input parameter
        """
        correct_values_dict = {
            "task_name": "test task",
            "ote_config": ConfigurableParameters("test header"),
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "task_name" parameter
            ("task_name", unexpected_int),
            # Unexpected integer is specified as "ote_config" parameter
            ("ote_config", unexpected_int),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=get_anomalib_config,
        )

    @e2e_pytest_unit
    def test_update_anomalib_config_params_validation(self):
        """
        <b>Description:</b>
        Check "update_anomalib_config" function parameters validation

        <b>Input data:</b>
        "update_anomalib_config" function input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "update_anomalib_config" function input parameter
        """
        correct_values_dict = {
            "anomalib_config": DictConfig({"some": "data"}),
            "ote_config": ConfigurableParameters("test header"),
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "anomalib_config" parameter
            ("anomalib_config", unexpected_str),
            # Unexpected string is specified as "ote_config" parameter
            ("ote_config", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=update_anomalib_config,
        )
