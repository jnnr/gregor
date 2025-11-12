from numpy.testing import assert_array_equal

from gregor.disaggregate import get_belongs_to_matrix

from fixtures import (
    dummy_raster,
    square_segmentation_2x2,
    square_segmentation_3x3,
    polygon_segmentation,
)


def test_belongs_to_matrix_square_segmentation_2x2(dummy_raster, square_segmentation_2x2):
    expected = [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ]

    belongs_to = get_belongs_to_matrix(dummy_raster, square_segmentation_2x2.geometry, nodata=-1)

    assert_array_equal(belongs_to, expected)


def test_belongs_to_matrix_square_segmentation_3x3(dummy_raster, square_segmentation_3x3):
    expected = [
        [-1, 0, 1, 2],
        [-1, 3, 4, 5],
        [-1, 6, 7, 8],
        [-1, 6, 7, 8],
    ]
    belongs_to = get_belongs_to_matrix(dummy_raster, square_segmentation_3x3.geometry, nodata=-1)

    assert_array_equal(belongs_to, expected)




def test_belongs_to_matrix_polygon_segmentation(dummy_raster, polygon_segmentation):
    expected = [
        [1, -1, -1, -1],
        [1,  1,  1, -1],
        [1,  1,  1,  0],
        [1,  0,  0,  0],
    ]

    belongs_to = get_belongs_to_matrix(dummy_raster, polygon_segmentation.geometry, nodata=-1)

    assert_array_equal(belongs_to, expected)
