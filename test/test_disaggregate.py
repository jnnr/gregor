import geopandas as gpd
import numpy as np
import pytest
import rioxarray as rxr
from gregor.disaggregate import (
    disaggregate_polygon_to_point,
    disaggregate_polygon_to_raster,
)


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
    return gpd.read_file("test/_files/points.geojson").set_index("id")


def test_disaggregate_2x2(dummy_raster, square_segmentation_2x2):
    data = square_segmentation_2x2

    data["value"] = [2, 2, 2, 2]

    expected = [
        [0.72727273, 0.0, 0.0, 0.0],
        [0.54545455, 0.72727273, 1.0, 1.0],
        [2.0, 0.0, 0.25, 0.75],
        [0.0, 0.0, 0.25, 0.75],
    ]

    disaggregated = disaggregate_polygon_to_raster(
        data=data, column="value", proxy=dummy_raster
    )

    assert (
        disaggregated.coarsen(x=2, y=2).sum()["value"].values == [[2, 2], [2, 2]]
    ).all()

    assert np.allclose(disaggregated["value"].values, expected)


def test_disaggregate_3x3():
    pass


def test_dissagregate_to_point(square_segmentation_2x2, points):
    data = square_segmentation_2x2

    data["value"] = [1, 3, 5, 7]

    data = data.drop(index=3)  # Drop the polygon with value 7 which has no points

    disaggregated = disaggregate_polygon_to_point(
        data=data, column="value", proxy=points, proxy_column="data"
    )

    assert disaggregated["disaggregated"].sum() == 9
