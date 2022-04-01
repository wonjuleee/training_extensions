# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from ote_anomalib.logging.logging import get_logger
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


class TestLoggingInputParamsValidation:
    @e2e_pytest_unit
    def test_get_logger_params_validation(self):
        """
        <b>Description:</b>
        Check "get_logger" function parameters validation

        <b>Input data:</b>
        "get_logger" function input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "get_logger" function input parameter
        """
        correct_values_dict = {
            "name": "test logger",
        }
        unexpected_float = 0.1
        unexpected_values = [
            # Unexpected float is specified as "name" parameter
            ("name", unexpected_float),
            # Unexpected float is specified as "log_file" parameter
            ("log_file", unexpected_float),
            # Unexpected float is specified as "log_level" parameter
            ("log_level", unexpected_float),
            # Unexpected float is specified as "file_mode" parameter
            ("file_mode", unexpected_float),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=get_logger,
        )
