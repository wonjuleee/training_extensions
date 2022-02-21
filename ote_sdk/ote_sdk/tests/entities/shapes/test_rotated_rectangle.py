# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import pytest

from ote_sdk.entities.shapes.polygon import Point
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.shapes.rotated_rectangle import RotatedRectangle
from ote_sdk.entities.shapes.shape import ShapeType
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements
from ote_sdk.utils.time_utils import now


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestRotatedRectangle:
    modification_date = now()

    def points(self):
        point1 = Point(0.5, 0.25)
        point2 = Point(0.75, 0.5)
        point3 = Point(0.5, 0.75)
        point4 = Point(0.25, 0.5)
        return [point1, point2, point3, point4]

    def rotated_1_points(self):
        point1, point2, point3, point4 = self.points()
        return [point2, point3, point4, point1]

    def rotated_2_points(self):
        point1, point2, point3, point4 = self.points()
        return [point3, point4, point1, point2]

    def rotated_3_points(self):
        point1, point2, point3, point4 = self.points()
        return [point4, point1, point2, point3]

    def other_points(self):
        point1 = Point(0.75, 0.25)
        point2 = Point(1.0, 0.5)
        point3 = Point(0.5, 1.0)
        point4 = Point(0.25, 0.75)
        return [point1, point2, point3, point4]

    def rotated_rectangle(self):
        return RotatedRectangle(self.points(), modification_date=self.modification_date)

    def rotated_1_rectangle(self):
        return RotatedRectangle(
            self.rotated_1_points(), modification_date=self.modification_date
        )

    def rotated_2_rectangle(self):
        return RotatedRectangle(
            self.rotated_2_points(), modification_date=self.modification_date
        )

    def rotated_3_rectangle(self):
        return RotatedRectangle(
            self.rotated_3_points(), modification_date=self.modification_date
        )

    def other_rotated_rectangle(self):
        return RotatedRectangle(
            self.other_points(), modification_date=self.modification_date
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_rotated_rectangle(self):
        """
        <b>Description:</b>
        Check RotatedRectangle parameters

        <b>Input data:</b>
        Points

        <b>Expected results:</b>
        Test passes if RotatedRectangle correctly calculates parameters and returns default values

        <b>Steps</b>
        1. Check RotatedRectangle params
        2. Check RotatedRectangle default values
        3. Check RotatedRectangle with empty points
        4. Check RotatedRectangle with invalid number of points
        """

        rotated_rectangle = self.rotated_rectangle()
        modification_date = self.modification_date
        assert len(rotated_rectangle.points) == 4
        assert rotated_rectangle.modification_date == modification_date
        assert rotated_rectangle.points == self.points()

        assert rotated_rectangle.type == ShapeType.ROTATED_RECTANGLE

        empty_points_list = []
        with pytest.raises(ValueError):
            RotatedRectangle(empty_points_list)

        with pytest.raises(ValueError):
            RotatedRectangle(self.points()[:3])

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_rotated_rectangle_magic_methods(self):
        """
        <b>Description:</b>
        Check RotatedRectangle __repr__, __eq__, __hash__ methods

        <b>Input data:</b>
        Initialized instance of RotatedRectangle

        <b>Expected results:</b>
        Test passes if RotatedRectangle magic methods returns correct values

        <b>Steps</b>
        1. Initialize RotatedRectangle instance
        2. Check returning value of magic methods
        """

        rotated_rectangle = self.rotated_rectangle()

        assert (
            "RotatedRectangle(points=[Point(0.5, 0.25), Point(0.75, 0.5), Point(0.5, 0.75), Point(0.25, 0.5)])"
            == repr(rotated_rectangle)
        )

        other_rotated_rectangle = self.rotated_rectangle()
        third_rotated_rectangle = self.other_rotated_rectangle()
        assert rotated_rectangle == other_rotated_rectangle
        assert rotated_rectangle != third_rotated_rectangle
        assert rotated_rectangle != str

        assert hash(rotated_rectangle) == hash(str(rotated_rectangle))

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_rotated_rectangle_normalize_wrt_roi_shape(self):
        """
        <b>Description:</b>
        Check RotatedRectangle normalize_wrt_roi_shape methods

        <b>Input data:</b>
        Initialized instance of RotatedRectangle
        Initialized instance of Rectangle

        <b>Expected results:</b>
        Test passes if RotatedRectangle normalize_wrt_roi_shape returns correct values

        <b>Steps</b>
        1. Initialize RotatedRectangle instance
        2. Check returning value
        """

        rotated_rectangle = self.rotated_rectangle()
        roi = Rectangle(x1=0.0, x2=0.5, y1=0.0, y2=0.5)
        normalized = rotated_rectangle.normalize_wrt_roi_shape(roi)

        assert normalized.points == [
            Point(0.25, 0.125),
            Point(0.375, 0.25),
            Point(0.25, 0.375),
            Point(0.125, 0.25),
        ]

        with pytest.raises(ValueError):
            rotated_rectangle.normalize_wrt_roi_shape("123")

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_rotated_rectangle_denormalize_wrt_roi_shape(self):
        """
        <b>Description:</b>
        Check RotatedRectangle denormalize_wrt_roi_shape methods

        <b>Input data:</b>
        Initialized instance of RotatedRectangle
        Initialized instance of Rectangle

        <b>Expected results:</b>
        Test passes if RotatedRectangle denormalize_wrt_roi_shape returns correct values

        <b>Steps</b>
        1. Initialize RotatedRectangle instance
        2. Check returning value
        """

        rotated_rectangle = RotatedRectangle(
            points=[
                Point(0.25, 0.125),
                Point(0.375, 0.25),
                Point(0.25, 0.375),
                Point(0.125, 0.25),
            ]
        )
        roi = Rectangle(x1=0.0, x2=0.5, y1=0.0, y2=0.5)
        denormalized = rotated_rectangle.denormalize_wrt_roi_shape(roi)
        assert denormalized.points == self.points()

        with pytest.raises(ValueError):
            rotated_rectangle.denormalize_wrt_roi_shape("123")

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_rotated_rectangle__as_shapely_polygon(self):
        """
        <b>Description:</b>
        Check RotatedRectangle _as_shapely_polygon methods

        <b>Input data:</b>
        Initialized instance of RotatedRectangle

        <b>Expected results:</b>
        Test passes if RotatedRectangle _as_shapely_polygon returns correct values

        <b>Steps</b>
        1. Initialize RotatedRectangle instance
        2. Check returning value
        """

        rotated_rectangle = self.rotated_rectangle()
        rotated_rectangle2 = self.other_rotated_rectangle()
        shapely_polygon = rotated_rectangle._as_shapely_polygon()
        shapely_polygon2 = rotated_rectangle2._as_shapely_polygon()
        assert shapely_polygon.area == pytest.approx(0.125)
        assert (
            str(shapely_polygon)
            == "POLYGON ((0.5 0.25, 0.75 0.5, 0.5 0.75, 0.25 0.5, 0.5 0.25))"
        )
        assert shapely_polygon != shapely_polygon2

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_rotated_rectangle_get_area(self):
        """
        <b>Description:</b>
        Check RotatedRectangle get_area method

        <b>Input data:</b>
        Instances of RotatedRectangle class

        <b>Expected results:</b>
        Test passes if get_area method returns expected value of RotatedRectangle area

        <b>Steps</b>
        1. Check get_area method for RotatedRectangle instance
        """
        for rectangle, expected_area in [
            (self.rotated_rectangle(), 0.125),
            (self.other_rotated_rectangle(), 0.25),
        ]:
            assert rectangle.get_area() == pytest.approx(expected_area)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_rotated_rectangle_get_orientation(self):
        """
        <b>Description:</b>
        Check RotatedRectangle get_orientation method

        <b>Input data:</b>
        Instances of RotatedRectangle class

        <b>Expected results:</b>
        Test passes if get_orientation method returns expected value of RotatedRectangle orientation

        <b>Steps</b>
        1. Check get_orientation method for RotatedRectangle instance
        """
        for rectangle, expected_orientation in [
            (self.rotated_rectangle(), Point(0.7071067811865475, -0.7071067811865475)),
            (self.rotated_1_rectangle(), Point(0.7071067811865475, 0.7071067811865475)),
            (
                self.rotated_2_rectangle(),
                Point(-0.7071067811865475, 0.7071067811865475),
            ),
            (
                self.rotated_3_rectangle(),
                Point(-0.7071067811865475, -0.7071067811865475),
            ),
            (
                self.other_rotated_rectangle(),
                Point(0.7071067811865475, -0.7071067811865475),
            ),
        ]:
            assert rectangle.get_orientation() == expected_orientation
