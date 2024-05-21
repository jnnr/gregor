import geopandas as gpd
import xarray as xr
import numpy as np
from rasterio.features import geometry_mask
from pathlib import Path


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
    elif resolution is not None and proxy is not None:
        raise ValueError("Only one of resolution or proxy should be provided.")
    elif resolution is not None and proxy is None:
        print("Disaggregating using resolution.")
        proxy = get_uniform_proxy(data.geometry, resolution)
    elif resolution is None and proxy is not None:
        print("Disaggregating using proxy.")
        assert proxy.rio.crs == data.geometry.crs, f"Proxy and data should have the same CRS. But proxy has {proxy.rio.crs} and data has {data.geometry.crs}."

    # TODO: Look at atlite's ExclusionContainer for inspiration on how to implement this.
    # probably implemented in shape_availability()
    # Each raster point belongs to one spatial_unit
    belongs_to = get_belongs_to_matrix(proxy, data.geometry)
    _data = data.to_xarray()
    _data = _data.drop_vars(["geometry"])
    normalization = belongs_to.sum(axis=(0, 1))

    # Remove regions that do not belong to any geometry
    normalization = normalization.drop_sel(id=normalization.id[normalization==0])
    belongs_to = belongs_to.sel(id=normalization.id)
    _data = _data.sel(id=normalization.id)

    # Disaggregate data to raster using proxy
    # raster_data_{x,y} = 1/normalization_{id} * _data_{id} * belongs_to_{id,x,y} * proxy_{x,y}
    raster_data = 1 / normalization * _data * belongs_to * proxy

    return raster_data


def get_uniform_proxy(spatial_units: gpd.GeoSeries, raster_resolution: tuple[int,int]) -> xr.Dataset:
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

    # TODO Set transform and crs
    # uniform_proxy = uniform_proxy.rio.set_spatial_dims('x', 'y')
    # uniform_proxy = uniform_proxy.rio.write_transform()
    uniform_proxy = uniform_proxy.rio.set_crs(spatial_units.crs)

    return uniform_proxy


def get_belongs_to_matrix(raster_data: xr.Dataset, spatial_units: gpd.GeoSeries) -> xr.Dataset:
    # create an empty dataarray with the coords matching raster_data and spatial_units
    belongs_to_matrix = xr.DataArray(data=None, dims=["y", "x", "id"], coords={"y": raster_data.y, "x": raster_data.x, "id": spatial_units.index})
    belongs_to_matrix.attrs['transform'] = raster_data.rio.transform
    belongs_to_matrix.attrs['crs'] = raster_data.rio.crs

    for id, geometry in spatial_units.items():
        mask = geometry_mask([geometry], out_shape=raster_data.shape, transform=raster_data.rio.transform(), invert=True)
        mask = xr.DataArray(mask, coords=raster_data.coords, dims=raster_data.dims)
        belongs_to_matrix.loc[:,:,id] = mask

    belongs_to_matrix = belongs_to_matrix.astype("bool")

    return belongs_to_matrix



