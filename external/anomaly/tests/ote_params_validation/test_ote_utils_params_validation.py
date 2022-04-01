# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest

from ote_anomalib.data.utils import split_local_global_dataset, split_local_global_resultset, contains_anomalous_images
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit


class TestOteUtilsInputParamsValidation:
    @e2e_pytest_unit
    def test_split_local_global_dataset_params_validation(self):
        """
        <b>Description:</b>
        Check "split_local_global_dataset" function parameters validation

        <b>Input data:</b>
        "dataset" non-DatasetEntity object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "split_local_global_dataset" function input parameter
        """
        with pytest.raises(ValueError):
            split_local_global_dataset(dataset="unexpected string")

    @e2e_pytest_unit
    def test_split_local_global_resultset_params_validation(self):
        """
        <b>Description:</b>
        Check "split_local_global_resultset" function parameters validation

        <b>Input data:</b>
        "resultset" non-ResultSetEntity object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "split_local_global_resultset" function input parameter
        """
        with pytest.raises(ValueError):
            split_local_global_resultset(resultset="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_contains_anomalous_images_params_validation(self):
        """
        <b>Description:</b>
        Check "contains_anomalous_images" function parameters validation

        <b>Input data:</b>
        "dataset" non-DatasetEntity object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "contains_anomalous_images" function input parameter
        """
        with pytest.raises(ValueError):
            contains_anomalous_images(dataset="unexpected string")  # type: ignore
