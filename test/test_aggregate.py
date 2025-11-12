import numpy as np
import pytest
from numpy.testing import assert_array_equal

from gregor.aggregate import aggregate_raster_to_polygon
from fixtures import (
    dummy_raster,
    square_segmentation_2x2,
    square_segmentation_3x3,
)


def test_agg_tif_2x2(square_segmentation_2x2):
    agg_raster_poly = aggregate_raster_to_polygon(
        "test/_files/raster.tif", square_segmentation_2x2
    )
    agg_raster_poly = agg_raster_poly["sum"].to_numpy().reshape(2, 2)
    expected = [
        [2.75, 1.0],
        [0.75, 2.0],
    ]

    assert_array_equal(agg_raster_poly, expected)


def test_agg_array_2x2(square_segmentation_2x2, dummy_raster):
    agg_raster_poly = aggregate_raster_to_polygon(dummy_raster, square_segmentation_2x2)
    agg_raster_poly = agg_raster_poly["sum"].to_numpy().reshape(2, 2)

    expected = [
        [2.75, 1.0],
        [0.75, 2.0],
    ]

    assert_array_equal(agg_raster_poly, expected)


@pytest.mark.skip(
    "rasterstats.zonal_stats does provide exact aggregation when pixels are not aligned with polygons. "
    "This is because a pixel can only belong or not belong to a polygon and not be split."
)
def test_agg_tif_3x3(square_segmentation_3x3):
    agg_raster_poly = aggregate_raster_to_polygon(
        "test/_files/raster.tif", square_segmentation_3x3
    )
    agg_raster_poly =  agg_raster_poly["sum"].to_numpy().reshape(3, 3)

    expected = [
        [2.50, 1.50, 1.00],
        [2.50, 1.75, 1.75],
        [0.75, 0.50, 2.00],
    ]

    assert_array_equal(agg_raster_poly, expected)


@pytest.mark.skip(
    "rasterstats.zonal_stats does provide exact aggregation when pixels are not aligned with polygons. "
    "This is because a pixel can only belong or not belong to a polygon and not be split."
)
def test_agg_array_3x3(square_segmentation_3x3, dummy_raster):
    agg_raster_poly = aggregate_raster_to_polygon(
        dummy_raster(), square_segmentation_3x3()
    )
    agg_raster_poly = agg_raster_poly["sum"].to_numpy().reshape(3, 3)

    expected = [
        [2.50, 1.50, 1.00],
        [2.50, 1.75, 1.75],
        [0.75, 0.50, 2.00],
    ]

    assert_array_equal(agg_raster_poly, expected)
