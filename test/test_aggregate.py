import numpy as np
import pytest
from numpy.testing import assert_array_equal

from gregor.aggregate import aggregate_raster_to_polygon
from fixtures import (
    dummy_raster,
    square_segmentation_2x2,
    square_segmentation_3x3,
)


@pytest.fixture
def expected_raster_to_2x2():
    expected = [
        [2.75, 1.0],
        [0.75, 2.0],
    ]
    return expected


@pytest.fixture
def expected_raster_to_3x3():
    expected = [
        [2.50, 1.50, 1.00],
        [2.50, 1.75, 1.75],
        [0.75, 0.50, 2.00],
    ]
    return expected


@pytest.mark.parametrize(
    'raster, polygons, expected', 
    [
        ('dummy_raster', 'square_segmentation_2x2', "expected_raster_to_2x2"),
        # ('dummy_raster', 'squares_3x3', "expected_raster_to_3x3"),
        # rasterstats.zonal_stats does provide exact aggregation when pixels are not aligned with polygons.
        # This is because a pixel can only belong or not belong to a polygon and not be split.
    ],
)
def test_raster_dataarray_to_poly(raster, polygons, expected, request):
    raster = request.getfixturevalue(raster)
    polygons = request.getfixturevalue(polygons)
    expected = request.getfixturevalue(expected)

    agg_raster_poly = aggregate_raster_to_polygon(raster, polygons)
    
    expected = (np.array(expected).flatten())

    assert_array_equal(
        agg_raster_poly["sum"].to_numpy(), 
        expected
    )


@pytest.mark.parametrize(
    'file, polygons, expected', 
    [
        ('test/_files/raster.tif', 'square_segmentation_2x2', "expected_raster_to_2x2"),
        # ('test/_files/raster.tif', 'squares_3x3', "expected_raster_to_3x3"),
        # rasterstats.zonal_stats does provide exact aggregation when pixels are not aligned with polygons.
        # This is because a pixel can only belong or not belong to a polygon and not be split.
    ],
)
def test_raster_file_to_poly(file, polygons, expected, request):
    polygons = request.getfixturevalue(polygons)
    expected = request.getfixturevalue(expected)
    agg_raster_poly = aggregate_raster_to_polygon(file, polygons)
    
    expected = (np.array(expected).flatten())

    assert_array_equal(
        agg_raster_poly["sum"].to_numpy(), 
        expected
    )
