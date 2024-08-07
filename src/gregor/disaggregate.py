import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio.features import geometry_mask


def disaggregate_polygon_to_raster(
    data: gpd.GeoDataFrame,
    column: str,
    proxy: xr.Dataset,
    to_data_crs: bool = False,
) -> xr.Dataset:
    r"""
    Disaggregate polygon data to raster data using proxy.
    Normalization of the proxy happens internally.

    Parameters
    ----------
    data : gpd.GeoDataFrame
        Data to be disaggregated.
    column : str
        Column name of the data to be disaggregated.
    proxy : xr.Dataset
        Proxy data for disaggregation.
    to_data_crs : bool, optional
        Whether to reproject proxy to `data`'s CRS or keep it in `raster`'s CRS. Default is False.

    Returns
    -------
    xr.Dataset
        Disaggregated raster data.
    """
    _data = data.copy()
    if not proxy.rio.crs == data.crs:
        print(
            f"CRS of `proxy` ({proxy.rio.crs}) does not match CRS of `data` ({data.crs}). Reprojecting CRS of `data` to `proxy`'s CRS."
        )
        _data = _data.to_crs(proxy.rio.crs)

    # Each raster point belongs to one spatial_unit
    belongs_to = get_belongs_to_matrix(proxy, _data.geometry)
    _data = _data[[column]].to_xarray()
    normalization = proxy.groupby(belongs_to).sum().rename(group="id")

    # # Remove regions that do not belong to any geometry
    _data = _data.sel(id=normalization.coords["id"])

    # Disaggregate data to raster using proxy
    # raster_data_{x,y} = 1/normalization_{id} * _data_{id} * belongs_to_{id,x,y} * proxy_{x,y}
    raster_data = xr.DataArray(
        data=0, dims=["y", "x"], coords={"y": proxy.y, "x": proxy.x}
    )
    for id in normalization.coords["id"]:
        raster_data_id = (
            1 / normalization.sel(id=id) * _data.sel(id=id) * (belongs_to == id) * proxy
        )
        raster_data = raster_data + raster_data_id

    if to_data_crs:
        print(f"Reprojecting results to `data`'s CRS {data.crs}.")
        raster_data = raster_data.rio.reproject(data.crs)

    return raster_data


def get_uniform_proxy(
    polygons: gpd.GeoSeries, raster_resolution: tuple[int, int]
) -> xr.Dataset:
    r"""
    Get a uniform proxy which sums to one for each region.

    Parameters
    ----------
    polygons : gpd.GeoSeries
        Polygons to compute the proxy for.
    raster_resolution : tuple[int, int]
        Resolution of the desired raster proxy.

    Returns
    -------
    xr.Dataset
        Uniform proxy which sums to 1 in each region.
    """
    # get spatial extent of spatial_units
    x_min, y_min, x_max, y_max = polygons.total_bounds

    # define coords
    x_coords = np.linspace(x_min, x_max, raster_resolution[0])
    y_coords = np.linspace(y_min, y_max, raster_resolution[1])

    # create raster Dataset
    uniform_proxy = xr.Dataset(
        data_vars={}, coords={"x": ("x", x_coords), "y": ("y", y_coords)}
    )

    # TODO Set transform and crs
    # uniform_proxy = uniform_proxy.rio.set_spatial_dims('x', 'y')
    # uniform_proxy = uniform_proxy.rio.write_transform()
    uniform_proxy = uniform_proxy.rio.set_crs(polygons.crs)

    return uniform_proxy


def get_belongs_to_matrix(
    raster_data: xr.Dataset, polygons: gpd.GeoSeries
) -> xr.Dataset:
    r"""
    Get a matrix which indicates which polygon each raster point belongs to.

    Parameters
    ----------
    raster_data : xr.Dataset
        Raster data to get the matrix for.
    polygons : gpd.GeoSeries
        Polygons to compute the matrix for.

    Returns
    -------
    xr.Dataset
        Matrix which indicates which polygon each raster point belongs to.
    """
    assert len(raster_data.dims) == 2, "Raster data should have 2 dimensions."
    # create an empty dataarray with the coords matching raster_data and spatial_units
    belongs_to_matrix = xr.DataArray(
        data=None, dims=["y", "x"], coords={"y": raster_data.y, "x": raster_data.x}
    )
    belongs_to_matrix.attrs["transform"] = raster_data.rio.transform
    belongs_to_matrix.attrs["crs"] = raster_data.rio.crs

    for id, geometry in polygons.items():
        mask = geometry_mask(
            [geometry],
            out_shape=raster_data.shape,
            transform=raster_data.rio.transform(),
            invert=True,
        )
        mask = xr.DataArray(mask, coords=raster_data.coords, dims=raster_data.dims)
        # assert belongs_to_matrix.where(mask).isnull().all(), "Trying to assign to value which is not None. Maybe cause of overlapping geometries."
        belongs_to_matrix = belongs_to_matrix.where(~mask, id)

    return belongs_to_matrix
