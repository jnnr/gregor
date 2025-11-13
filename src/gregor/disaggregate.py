import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.features import rasterize

from gregor.aggregate import aggregate_raster_to_polygon


def disaggregate_polygon_to_raster(
    data: gpd.GeoDataFrame,
    column: str,
    proxy: xr.DataArray,
    chunk_size: int = 1024,
    to_data_crs: bool = False,
) -> xr.DataArray:
    r"""
    Disaggregate polygon data to raster data using proxy.
    Normalization of the proxy happens internally.

    Parameters
    ----------
    data : gpd.GeoDataFrame
        Data to be disaggregated.
    column : str
        Name of the attribute in `data` to disaggregate.
    proxy : xr.DataArray
        Raster whose intensities act as weights.
    chunk_size : int, default 1024
        Square chunk edge length for the proxy and ID rasters.
    to_data_crs : bool, default False
        If True, reprojects the output raster back to `data`'s CRS.

    Returns
    -------
    xr.DataArray
        Disaggregated raster data.
    """
    if isinstance(proxy, xr.DataArray):
        _proxy = proxy
    elif isinstance(proxy, xr.Dataset):
        if len(proxy.data_vars) == 1:
            raise DeprecationWarning(
                "Passing DataSet is deprecated and will be disallowed in the future. Use DataArray instead."
            )
            var_name = next(iter(proxy.data_vars))
            _proxy = proxy[var_name]
        else:
            raise ValueError(
                f"Cannot compute multi-variable Dataset of length {len(proxy.data_vars)}. "
                "Pass a DataArray instead."
            )
    else:
        raise TypeError(
            f"`proxy` must be an xarray DataArray or Dataset, got {type(proxy)}."
        )

    # make an internal copy of data with numerical index
    _data = data.copy()
    _data = _data.reset_index(drop=True)  # need integer IDs

    # make sure that crs of data and proxy match
    if _proxy.rio.crs != _data.crs:
        print(
            f"CRS of `data` ({_data.crs}) does not match CRS of `proxy` ({_proxy.rio.crs}). Reprojecting CRS of `data` to match `proxy`'s CRS."
        )
        _data = _data.to_crs(_proxy.rio.crs)

    # convert _proxy to chunked DataArray in float32
    _proxy = _proxy.astype("float32").chunk({"y": chunk_size, "x": chunk_size})

    # prepare normalisation, which is the sum of proxy values inside a polygon
    normalisation = aggregate_raster_to_polygon(_proxy, _data, stats=["sum"])

    # set up look up array
    # allocate extra slot (nodata) that will later propagate NaN outside given geometries

    # data_values contains the original data defined on polygons and a nodata.
    data_values = _data[column].astype("float32").values
    data_values = np.append(data_values, np.nan)

    # normalisation_values contains the proxy, aggregated to polygons, and a 1.0 for nodata.
    normalisation_values = normalisation["sum"].astype("float32").values
    normalisation_values = np.append(normalisation_values, np.nan)

    value_lookup = data_values / normalisation_values

    # create belongs_to matrix and replace nodata with last index in value_lookup
    nodata = -1
    belongs_to = get_belongs_to_matrix(_proxy, _data.geometry, nodata=nodata)
    belongs_to = belongs_to.chunk(_proxy.chunks)
    
    id_nodata = len(value_lookup) - 1  # index of nodata in value_lookup
    belongs_to_safe = belongs_to.data
    belongs_to_safe = da.where(belongs_to_safe != nodata, belongs_to_safe, id_nodata).astype("int32")

    # lookup via map_blocks
    def lookup_func(blk):
        return value_lookup[blk]
    val_pix = da.map_blocks(lookup_func, belongs_to_safe, dtype="float32")

    # compute final raster data
    raster_data = da.where(belongs_to_safe != id_nodata, _proxy.data * val_pix, np.nan)

    raster = xr.DataArray(
        raster_data,
        dims=_proxy.dims,
        coords=_proxy.coords,
        name=column,
        attrs=_proxy.attrs,
    )

    if to_data_crs and _proxy.rio.crs != data.crs:
        print(f"Reprojecting results to `data`'s CRS {data.crs}.")
        raster = raster.rio.reproject(data.crs)

    return raster


def get_belongs_to_matrix(raster: xr.DataArray, polygons: gpd.GeoSeries, nodata: int=-1) -> xr.DataArray:
    r"""
    Get a matrix which indicates which polygon each raster point belongs to.

    Parameters
    ----------
    raster : xr.DataArray
        Raster array to get the matrix for.
    polygons : gpd.GeoSeries
        Polygons to compute the matrix for.
    nodata : int
        Value to use as NaN, i.e. for pixels that do not belong to any polygon.

    Returns
    -------
    xr.DataArray
        Matrix which indicates which polygon each raster point belongs to.
    """
    assert len(raster.dims) == 2, "Raster data should have 2 dimensions."

    shapes = [(geom, i) for i, geom in enumerate(polygons)]
    arr = rasterize(
        shapes,
        out_shape=raster.shape,
        transform=raster.rio.transform(),
        fill=nodata,  # fills invalid cases
        dtype="int32",
    )

    return xr.DataArray(arr, coords=raster.coords, dims=raster.dims)


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


def disaggregate_polygon_to_point(
    data: gpd.GeoDataFrame,
    column: str,
    proxy: gpd.GeoDataFrame,
    proxy_column: str,
    to_data_crs: bool = False,
) -> gpd.GeoDataFrame:
    r"""
    Disaggregate polygon data to point data using proxy.

    Parameters
    ----------
    data : gpd.GeoDataFrame
        Data to be disaggregated.
    column : str
        Column name of the data to be disaggregated.
    proxy : gpd.GeoSeries
        Proxy data with point geometries for disaggregation.
    proxy_column : str
        Column name of the proxy data.
    to_data_crs : bool, optional
        Whether to reproject proxy to `data`'s CRS or keep it in `raster`'s CRS. Default is False.
    """
    _data = data.copy()
    points = proxy.copy()

    # compare crs. If not the same, project data to proxy's crs
    if not proxy.crs == _data.crs:
        print(
            f"CRS of `proxy` ({proxy.crs}) does not match CRS of `data` ({_data.crs}). Reprojecting CRS of `data` to `proxy`'s CRS."
        )
        _data = _data.to_crs(proxy.crs)

    # Find out which polygon each point belongs to.
    points["belongs_to"] = points.geometry.apply(
        lambda point: _data.index[_data.contains(point)]
    )

    # Make sure that it belongs to only one polygon
    assert points.belongs_to.apply(len).max() == 1, (
        "Every Point should belong to exactly one polygon."
    )
    points.belongs_to = points.belongs_to.apply(lambda x: x[0])

    # Warn if there are polygons without points.
    polygons_without_points = set(set(data.index)).difference(points.belongs_to)
    if polygons_without_points:
        raise Warning(
            f"These polygons have no points to disaggregate to {polygons_without_points}."
        )

    # normalization_polygon is the sum of `proxy_column` over all points that belong to the polygon.
    normalization = points[["belongs_to", proxy_column]].groupby("belongs_to").sum()
    normalization = normalization.rename(columns={proxy_column: f"sum_{proxy_column}"})

    # disaggregated value for each point is column_proxy * column_polygon / normalization_polygon
    points = pd.merge(points, data[column], left_on="belongs_to", right_index=True)
    points = points.join(normalization, on="belongs_to")
    points["disaggregated"] = (
        points[column] * points[proxy_column] / points[f"sum_{proxy_column}"]
    )

    # Keep only the necessary columns
    points = points[["geometry", "disaggregated"]]

    if to_data_crs:
        print(f"Reprojecting results to `data`'s CRS {_data.crs}.")
        points = points.to_crs(_data.crs)

    return points
