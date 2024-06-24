import geopandas as gpd
import numpy as np
import pytest
import rioxarray as rxr
from gregor.aggregate import aggregate_raster_to_polygon


@pytest.fixture
def dummy_raster():
    return rxr.open_rasterio("test/_files/raster.tif").squeeze(drop=True)


@pytest.fixture
def square_segmentation_2x2():
    return gpd.read_file("test/_files/segmentation_2x2.geojson").set_index("id")


@pytest.fixture
def square_segmentation_3x3():
    return gpd.read_file("test/_files/segmentation_3x3.geojson").set_index("id")


@pytest.fixture
def points():
    return gpd.read_file("test/_files/points.geojson")


def test_agg_tif_2x2(square_segmentation_2x2):
    agg_raster_poly = aggregate_raster_to_polygon(
        "test/_files/raster.tif", square_segmentation_2x2
    )

    expected = [
        [2.75, 1.0],
        [0.75, 2.0],
    ]

    assert (
        np.rot90(agg_raster_poly["sum"].to_numpy().reshape(2, 2), k=1) == expected
    ).all()


def test_agg_array_2x2(square_segmentation_2x2, dummy_raster):
    agg_raster_poly = aggregate_raster_to_polygon(dummy_raster, square_segmentation_2x2)

    expected = [
        [2.75, 1.0],
        [0.75, 2.0],
    ]

    assert (
        np.rot90(agg_raster_poly["sum"].to_numpy().reshape(2, 2), k=1) == expected
    ).all()


@pytest.mark.skip(
    "rasterstats.zonal_stats does provide exact aggregation when pixels are not aligned with polygons. "
    "This is because a pixel can only belong or not belong to a polygon and not be split."
)
def test_agg_tif_3x3(square_segmentation_3x3):
    agg_raster_poly = aggregate_raster_to_polygon(
        "test/_files/raster.tif", square_segmentation_3x3
    )

    expected = [
        [2.50, 1.50, 1.00],
        [2.50, 1.75, 1.75],
        [0.75, 0.50, 2.00],
    ]

    assert (
        np.rot90(agg_raster_poly["sum"].to_numpy().reshape(3, 3), k=1)
        == np.array(expected)
    ).all()


@pytest.mark.skip(
    "rasterstats.zonal_stats does provide exact aggregation when pixels are not aligned with polygons. "
    "This is because a pixel can only belong or not belong to a polygon and not be split."
)
def test_agg_array_3x3(square_segmentation_3x3, dummy_raster):
    agg_raster_poly = aggregate_raster_to_polygon(
        dummy_raster(), square_segmentation_3x3()
    )

    expected = [
        [2.50, 1.50, 1.00],
        [2.50, 1.75, 1.75],
        [0.75, 0.50, 2.00],
    ]

    assert (
        np.rot90(agg_raster_poly["sum"].to_numpy().reshape(3, 3), k=1)
        == np.array(expected)
    ).all()
