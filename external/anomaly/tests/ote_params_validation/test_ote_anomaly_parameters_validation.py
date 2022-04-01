# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np

from ote_anomalib.exportable_code.anomaly_classification import AnomalyClassification
from ote_anomalib.exportable_code.anomaly_detection import AnomalyDetection
from ote_anomalib.exportable_code.anomaly_segmentation import AnomalySegmentation
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


class MockAnomalyClassification(AnomalyClassification):
    def __init__(self):
        pass


class MockAnomalyDetection(AnomalyDetection):
    def __init__(self):
        pass


class MockAnomalySegmentation(AnomalySegmentation):
    def __init__(self):
        pass


class TestAnomalyClassificationInputParamsValidation:
    @e2e_pytest_unit
    def test_anomaly_classification_postprocess_params_validation(self):
        """
        <b>Description:</b>
        Check AnomalyClassification class "postprocess" method parameters validation

        <b>Input data:</b>
        AnomalyClassification object, "outputs" unexpected type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "postprocess" method input parameter
        """
        random_array = np.random.randint(low=0, high=255, size=(3, 3, 3))
        anomaly_classification = MockAnomalyClassification()
        correct_values_dict = {
            "outputs": {"output_1": random_array},
            "meta": {"meta_1": "metadata"},
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "outputs" parameter
            ("outputs", unexpected_int),
            # Unexpected integer is specified as "outputs" dictionary key
            ("outputs", {unexpected_int: random_array}),
            # Unexpected integer is specified as "outputs" dictionary value
            ("outputs", {"output_1": unexpected_int}),
            # Unexpected integer is specified as "meta" parameter
            ("meta", unexpected_int),
            # Unexpected integer is specified as "meta" dictionary key
            ("meta", {unexpected_int: "metadata"}),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=anomaly_classification.postprocess,
        )


class TestAnomalyDetectionInputParamsValidation:
    @e2e_pytest_unit
    def test_anomaly_detection_postprocess_params_validation(self):
        """
        <b>Description:</b>
        Check AnomalyDetection class "postprocess" method parameters validation

        <b>Input data:</b>
        AnomalyDetection object, "outputs" unexpected type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "postprocess" method input parameter
        """
        random_array = np.random.randint(low=0, high=255, size=(3, 3, 3))
        anomaly_detection = MockAnomalyDetection()
        correct_values_dict = {
            "outputs": {"output_1": random_array},
            "meta": {"meta_1": "metadata"},
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "outputs" parameter
            ("outputs", unexpected_int),
            # Unexpected integer is specified as "outputs" dictionary key
            ("outputs", {unexpected_int: random_array}),
            # Unexpected integer is specified as "outputs" dictionary value
            ("outputs", {"output_1": unexpected_int}),
            # Unexpected integer is specified as "meta" parameter
            ("meta", unexpected_int),
            # Unexpected integer is specified as "meta" dictionary key
            ("meta", {unexpected_int: "metadata"}),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=anomaly_detection.postprocess,
        )


class TestAnomalySegmentationInputParamsValidation:
    @e2e_pytest_unit
    def test_anomaly_segmentation_postprocess_params_validation(self):
        """
        <b>Description:</b>
        Check AnomalySegmentation class "postprocess" method parameters validation

        <b>Input data:</b>
        AnomalySegmentation object, "outputs" unexpected type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "postprocess" method input parameter
        """
        random_array = np.random.randint(low=0, high=255, size=(3, 3, 3))
        anomaly_segmentation = MockAnomalySegmentation()
        correct_values_dict = {
            "outputs": {"output_1": random_array},
            "meta": {"meta_1": "metadata"},
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "outputs" parameter
            ("outputs", unexpected_int),
            # Unexpected integer is specified as "outputs" dictionary key
            ("outputs", {unexpected_int: random_array}),
            # Unexpected integer is specified as "outputs" dictionary value
            ("outputs", {"output_1": unexpected_int}),
            # Unexpected integer is specified as "meta" parameter
            ("meta", unexpected_int),
            # Unexpected integer is specified as "meta" dictionary key
            ("meta", {unexpected_int: "metadata"}),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=anomaly_segmentation.postprocess,
        )
