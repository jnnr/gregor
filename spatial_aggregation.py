import geopandas as gpd
import xarray as xr
import numpy as np

def disaggregate_polygon_to_raster(
        data: gpd.GeoDataFrame,
        crs: str,
        resolution: int=None,
        proxy: xr.Dataset=None,
    ) -> xr.Dataset:
    r"""
    Disaggregate polygon data to raster data using proxy or uniform density.
    """
    # Proxy for each region should add to one
    if resolution is None and proxy is None:
        raise ValueError("Either resolution or proxy should be provided.")
    if resolution is not None and proxy is not None:
        raise ValueError("Only one of resolution or proxy should be provided.")
    if resolution is not None:
        print("Disaggregating using resolution.")
        proxy = get_uniform_proxy(data.geometry, resolution)
        pass
    elif proxy is not None:
        print("Disaggregating using proxy.")
        assert proxy.rio.crs == data.geometry.crs, f"Proxy and data should have the same CRS. But proxy has {proxy.rio.crs} and data has {data.geometry.crs}."
    
    print(proxy)

    # TODO: Look at atlite's ExclusionContainer for inspiration on how to implement this.
    # probably implemented in shape_availability()
    # Each raster point belongs to one spatial_unit
    indicatormatrix = None
    _data = data.to_xarray()

    # raster_data = proxy * matrix_of_belongs_to(pixel, id_of_unit) * data
    raster_data = proxy * indicatormatrix * _data
    # Disaggregate data to raster using proxy

    raster_data = xr.Dataset()
    return raster_data


def get_uniform_proxy(spatial_units: gpd.GeoSeries, raster_resolution) -> xr.Dataset:
    r"""
    Get a uniform proxy for each region.
    """
    # get spatial extent of spatial_units
    x_min, y_min, x_max, y_max = spatial_units.total_bounds

    # define coords
    x_coords = np.linspace(x_min, x_max, raster_resolution[0])
    y_coords = np.linspace(y_min, y_max, raster_resolution[1])

    # create raster Dataset
    uniform_proxy = xr.Dataset(
        data_vars={},
        coords={'x': ('x', x_coords), 'y': ('y', y_coords)}
    )

    return uniform_proxy


def aggregate_raster_to_polygon(
        raster_data: xr.Dataset,
        crs: str,
        spatial_units: gpd.GeoSeries,
) -> gpd.GeoDataFrame:
    r"""
    Aggregate raster data with spatial units and aggregate the data.
    """
    # Aggregate raster data to spatial units
    

    return 
    