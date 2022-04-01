# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from ote_anomalib.data.mvtec import OteMvtecDataset
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


class TestOteMvtecDatasetInputParamsValidation:
    @e2e_pytest_unit
    def test_ote_mvtec_dataset_init_params_validation(self):
        """
        <b>Description:</b>
        Check OteMvtecDataset object initialization parameters validation

        <b>Input data:</b>
        OteMvtecDataset object initialization parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OteMvtecDataset object initialization parameter
        """
        correct_values_dict = {
            "path": "./some_path",
        }
        unexpected_dict = {"unexpected": "dict"}
        unexpected_values = [
            # Unexpected dictionary is specified as "path" parameter
            ("path", unexpected_dict),
            # Empty string is specified as "work_dir" parameter
            ("path", ""),
            # String with null-character is specified as "work_dir" parameter
            ("path", "null\0character/path"),
            # String with non-printable character is specified as "work_dir" parameter
            ("path", "\non_printable_character/path"),
            # Unexpected dictionary is specified as "split_ratio" parameter
            ("split_ratio", unexpected_dict),
            # Unexpected dictionary is specified as "seed" parameter
            ("seed", unexpected_dict),
            # Unexpected dictionary is specified as "create_validation_set" parameter
            ("create_validation_set", unexpected_dict),
            # Unexpected dictionary is specified as "task_type" parameter
            ("task_type", unexpected_dict),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=OteMvtecDataset,
        )
