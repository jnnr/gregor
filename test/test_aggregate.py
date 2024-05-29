import geopandas as gpd
import rioxarray as rxr
import pytest
import numpy as np
import pandas as pd
from spatial_disaggregation.aggregate import aggregate_raster_to_polygon


# @pytest.fixture
def dummy_raster():
    return rxr.open_rasterio("test/_files/raster.tif").squeeze(drop=True)


# @pytest.fixture
def square_segmentation_2x2():
    return gpd.read_file("test/_files/segmentation_2x2.geojson").set_index("id")

# @pytest.fixture
def square_segmentation_3x3():
    return gpd.read_file("test/_files/segmentation_3x3.geojson").set_index("id")

# @pytest.fixture
def points():
    return gpd.read_file("test/_files/points.geojson")


def test_agg_tif_2x2(dummy_raster):
    agg_raster_poly = aggregate_raster_to_polygon("test/_files/raster.tif", square_segmentation_2x2())
    agg_raster_poly = pd.DataFrame(agg_raster_poly, index=square_segmentation_2x2().index)
    agg_raster_poly.index.name = square_segmentation_2x2().index.name

    expected = [
        [2.75, 1.],
        [0.75, 2.],
    ]

    assert (np.rot90(agg_raster_poly["value_sum"].to_numpy().reshape(2, 2), k=1) == expected).all()


def test_agg_array_2x2(square_segmentation_2x2, dummy_raster):
    # from rasterstats import zonal_stats
    # import pandas as pd
    # import numpy as np
    # agg_raster_poly = zonal_stats(
    #     square_segmentation_2x2(), dummy_raster().values, affine=dummy_raster().rio.transform(), stats='sum', nodata=-999
    # )
    # agg_raster_poly = pd.DataFrame(agg_raster_poly, index=square_segmentation_2x2().index)
    # agg_raster_poly.index.name = square_segmentation_2x2().index.name
    agg_raster_poly = aggregate_raster_to_polygon("test/_files/raster.tif", square_segmentation_2x2())
    agg_raster_poly = pd.DataFrame(agg_raster_poly, index=square_segmentation_2x2().index)
    agg_raster_poly.index.name = square_segmentation_2x2().index.name

    expected = [
        [2.75, 1.],
        [0.75, 2.],
    ]

    assert (np.rot90(agg_raster_poly["value_sum"].to_numpy().reshape(2, 2), k=1) == expected).all()


def test_agg_tif_3x3(dummy_raster):
    agg_raster_poly = aggregate_raster_to_polygon("test/_files/raster.tif", square_segmentation_3x3())
    agg_raster_poly = pd.DataFrame(agg_raster_poly, index=square_segmentation_3x3().index)
    agg_raster_poly.index.name = square_segmentation_3x3().index.name
    print(np.rot90(agg_raster_poly["value_sum"].to_numpy().reshape(3, 3), k=1))


def test_agg_array_3x3(square_segmentation_3x3, dummy_raster):
    from rasterstats import zonal_stats
    import pandas as pd
    import numpy as np
    array = np.array(
        [   
            [7, 5, 1, 0.25, 0.25],
            [0, 5, 0.25, 0.25, 0.25],
            [1, 0.25, 0.25, 0.25, 0.25],
            [0.5, 0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25, 0.25],
        ]
    )
    agg_raster_poly = zonal_stats(
        square_segmentation_4x4(),
        array,
        affine=dummy_raster().rio.transform(),
        stats='sum',
        nodata=-999
    )
    agg_raster_poly = pd.DataFrame(agg_raster_poly, index=square_segmentation_3x3().index)
    agg_raster_poly.index.name = square_segmentation_3x3().index.name

    expected = [
        [0.25, 0.50, 0.50, 0.75],
        [0.50, 0.75, 1.25, 0.75],
        [0.50, 0.75, 0.75, 1.25],
        [0.75, 1.00, 1.00, 0.25],
    ]
    print(np.rot90(agg_raster_poly["sum"].to_numpy().reshape(4, 4), k=1))
    # assert (np.rot90(agg_raster_poly["sum"].to_numpy().reshape(4, 4), k=1) == np.array(expected)).all()

if __name__ == "__main__":
    test_agg_tif_2x2(dummy_raster)
    test_agg_array_2x2(square_segmentation_2x2, dummy_raster)
    # test_agg_tif_3x3(dummy_raster)
    # test_agg_array_3x3(square_segmentation_3x3, dummy_raster)
