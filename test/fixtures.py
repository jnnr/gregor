import pytest
import geopandas as gpd
import rioxarray as rxr


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
