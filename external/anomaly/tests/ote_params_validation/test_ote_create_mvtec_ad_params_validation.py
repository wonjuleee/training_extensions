# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest

from ote_anomalib.data.create_mvtec_ad_json_annotations import (
    create_bboxes_from_mask,
    create_polygons_from_mask,
    create_classification_json_items,
    create_detection_json_items,
    create_segmentation_json_items,
    save_json_items,
    create_task_annotations,
    create_mvtec_ad_category_annotations,
    create_mvtec_ad_annotations,
)
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


class TestCreateMVTecAdInputParamsValidation:
    @e2e_pytest_unit
    def test_create_bboxes_from_mask_params_validation(self):
        """
        <b>Description:</b>
        Check "create_bboxes_from_mask" function parameters validation

        <b>Input data:</b>
        "create_bboxes_from_mask" function input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "create_bboxes_from_mask" function input parameter
        """
        for unexpected_value in [
            # Unexpected integer is specified as "mask_path" parameter
            1,
            # Empty string is specified as "mask_path" parameter
            "",
            # Path to file with unexpected extension is specified as "mask_path" parameter
            "unexpected_extension.json",
            # Path to non-existing file is specified as "mask_path" parameter
            "./non_existing.jpg",
            # Path with null character is specified as "mask_path" parameter
            "null\0char.jpg",
            # Path with non-printable character is specified as "mask_path" parameter
            "\non_printable_char.jpg",
        ]:
            with pytest.raises(ValueError):
                create_bboxes_from_mask(mask_path=unexpected_value)

    @e2e_pytest_unit
    def test_create_polygons_from_mask_params_validation(self):
        """
        <b>Description:</b>
        Check "create_polygons_from_mask" function parameters validation

        <b>Input data:</b>
        "create_polygons_from_mask" function input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "create_polygons_from_mask" function input parameter
        """
        for unexpected_value in [
            # Unexpected integer is specified as "mask_path" parameter
            1,
            # Empty string is specified as "mask_path" parameter
            "",
            # Path to file with unexpected extension is specified as "mask_path" parameter
            "unexpected_extension.json",
            # Path to non-existing file is specified as "mask_path" parameter
            "./non_existing.jpg",
            # Path with null character is specified as "mask_path" parameter
            "null\0char.jpg",
            # Path with non-printable character is specified as "mask_path" parameter
            "\non_printable_char.jpg",
        ]:
            with pytest.raises(ValueError):
                create_polygons_from_mask(mask_path=unexpected_value)

    @e2e_pytest_unit
    def test_create_classification_json_items_params_validation(self):
        """
        <b>Description:</b>
        Check "create_classification_json_items" function parameters validation

        <b>Input data:</b>
        "pd_items" non-pd.DataFrame object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "create_classification_json_items" function input parameter
        """
        with pytest.raises(ValueError):
            create_classification_json_items(pd_items="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_create_detection_json_items_params_validation(self):
        """
        <b>Description:</b>
        Check "create_detection_json_items" function parameters validation

        <b>Input data:</b>
        "pd_items" non-pd.DataFrame object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "create_detection_json_items" function input parameter
        """
        with pytest.raises(ValueError):
            create_detection_json_items(pd_items="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_create_segmentation_json_items_params_validation(self):
        """
        <b>Description:</b>
        Check "create_segmentation_json_items" function parameters validation

        <b>Input data:</b>
        "pd_items" non-pd.DataFrame object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "create_segmentation_json_items" function input parameter
        """
        with pytest.raises(ValueError):
            create_segmentation_json_items(pd_items="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_save_json_items_params_validation(self):
        """
        <b>Description:</b>
        Check "save_json_items" function parameters validation

        <b>Input data:</b>
        "save_json_items" function input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "save_json_items" function input parameter
        """
        correct_values_dict = {
            "json_items": {"json_item_1": "some data"},
            "file": "./file.json",
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "json_items" parameter
            ("json_items", unexpected_int),
            # Unexpected integer is specified as "json_items" parameter dictionary key
            ("json_items", {unexpected_int: "some data"}),
            # Unexpected integer is specified as "file" parameter
            ("file", unexpected_int),
            # Empty string is specified as "file" parameter
            ("file", ""),
            # Path with null character is specified as "file" parameter
            ("file", "null\0char.json"),
            # Path with non-printable character is specified as "file" parameter
            ("file", "\non_printable_char.json"),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=save_json_items,
        )

    @e2e_pytest_unit
    def test_create_task_annotations_params_validation(self):
        """
        <b>Description:</b>
        Check "create_task_annotations" function parameters validation

        <b>Input data:</b>
        "create_task_annotations" function input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "create_task_annotations" function input parameter
        """
        correct_values_dict = {
            "task": "classification",
            "data_path": "./data/path",
            "annotation_path": "./annotation.json",
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "task" parameter
            ("task", unexpected_int),
            # Unexpected integer is specified as "data_path" parameter
            ("data_path", unexpected_int),
            # Empty string is specified as "data_path" parameter
            ("data_path", ""),
            # Path with null character is specified as "data_path" parameter
            ("data_path", "./null\0char"),
            # Path with non-printable character is specified as "data_path" parameter
            ("data_path", "./\non_printable"),
            # Unexpected integer is specified as "annotation_path" parameter
            ("annotation_path", unexpected_int),
            # Empty string is specified as "annotation_path" parameter
            ("annotation_path", ""),
            # Path with null character is specified as "annotation_path" parameter
            ("annotation_path", "null\0char.json"),
            # Path with non-printable character is specified as "annotation_path" parameter
            ("annotation_path", "\non_printable_char.json"),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=create_task_annotations,
        )

    @e2e_pytest_unit
    def test_create_mvtec_ad_category_annotations_params_validation(self):
        """
        <b>Description:</b>
        Check "create_mvtec_ad_category_annotations" function parameters validation

        <b>Input data:</b>
        "create_mvtec_ad_category_annotations" function input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "create_mvtec_ad_category_annotations" function input parameter
        """
        correct_values_dict = {
            "data_path": "./data/path",
            "annotation_path": "./annotation.json",
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "data_path" parameter
            ("data_path", unexpected_int),
            # Empty string is specified as "data_path" parameter
            ("data_path", ""),
            # Path with null character is specified as "data_path" parameter
            ("data_path", "./null\0char"),
            # Path with non-printable character is specified as "data_path" parameter
            ("data_path", "./\non_printable"),
            # Unexpected integer is specified as "annotation_path" parameter
            ("annotation_path", unexpected_int),
            # Empty string is specified as "annotation_path" parameter
            ("annotation_path", ""),
            # Path with null character is specified as "annotation_path" parameter
            ("annotation_path", "null\0char.json"),
            # Path with non-printable character is specified as "annotation_path" parameter
            ("annotation_path", "\non_printable_char.json"),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=create_mvtec_ad_category_annotations,
        )

    @e2e_pytest_unit
    def test_create_mvtec_ad_annotations_params_validation(self):
        """
        <b>Description:</b>
        Check "create_mvtec_ad_annotations" function parameters validation

        <b>Input data:</b>
        "create_mvtec_ad_annotations" function input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        "create_mvtec_ad_annotations" function input parameter
        """
        correct_values_dict = {
            "mvtec_data_path": "./data/path",
            "annotation_path": "./annotation.json",
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "mvtec_data_path" parameter
            ("mvtec_data_path", unexpected_int),
            # Empty string is specified as "mvtec_data_path" parameter
            ("mvtec_data_path", ""),
            # Path with null character is specified as "mvtec_data_path" parameter
            ("mvtec_data_path", "./null\0char"),
            # Path with non-printable character is specified as "mvtec_data_path" parameter
            ("mvtec_data_path", "./\non_printable"),
            # Unexpected integer is specified as "mvtec_annotation_path" parameter
            ("mvtec_annotation_path", unexpected_int),
            # Empty string is specified as "mvtec_annotation_path" parameter
            ("mvtec_annotation_path", ""),
            # Path with null character is specified as "mvtec_annotation_path" parameter
            ("mvtec_annotation_path", "null\0char.json"),
            # Path with non-printable character is specified as "mvtec_annotation_path" parameter
            ("mvtec_annotation_path", "\non_printable_char.json"),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=create_mvtec_ad_annotations,
        )
