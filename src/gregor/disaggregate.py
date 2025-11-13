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

    # make sure that index has a name in internal copy of 'data'
    _data = data.copy()
    index_name = _data.index.name or "id"
    _data.index.name = index_name

    # make sure that crs of data and proxy match
    if _proxy.rio.crs != _data.crs:
        print(
            f"CRS of `proxy` ({_proxy.rio.crs}) does not match CRS of `data` ({data.crs}). Reprojecting CRS of `data` to `proxy`'s CRS."
        )
        _data = _data.to_crs(_proxy.rio.crs)

    # convert _proxy to chunked DataArray in float32
    _proxy = _proxy.astype("float32").chunk({"y": chunk_size, "x": chunk_size})

    # create belongs_to matrix with integer ids
    if not np.issubdtype(_data.index.dtype, np.integer):
        print("Non-integer index detected, resetting to numeric for processing.")
        geom_id_source = _data.reset_index(drop=True).geometry  # integer IDs
    else:
        geom_id_source = _data.geometry

    belongs_to = get_belongs_to_matrix(_proxy, geom_id_source, nodata=-1)
    belongs_to = belongs_to.chunk(_proxy.chunks)

    # prepare normalisation, which is the sum of proxy values inside a polygon
    normalisation = aggregate_raster_to_polygon(_proxy, _data, stats=["sum"])

    # set up arrays
    # +1 extra slot (sentinel) that will later propagate NaN outside given geometries
    max_id = int(belongs_to.max().compute())  # get the 'true' maximum
    n_ids = max_id + 1  # valid IDs: 0 … max_id
    id_sentinel = n_ids  # index of zero‑filled slot

    # val_full contains the original data defined on polygons and a nodata
    val_full = np.zeros(n_ids + 1, dtype="float32")
    val_full[id_sentinel] = np.nan
    val_full[: len(_data)] = _data[column].astype("float32").values

    # norm_full contains the proxy, aggregated to polygons, and a 1.0 for nodata
    norm_full = np.zeros(n_ids + 1, dtype="float32")
    norm_full[id_sentinel] = 1.0  # avoid 0/0 inside map_blocks
    norm_full[:n_ids] = normalisation["sum"].values

    # raster assembly
    ids = belongs_to.data  # (y,x)
    ids_safe = da.where(ids >= 0, ids, id_sentinel).astype(
        "int32"
    )  # max > billion polygons

    # lookup via map_blocks, works with N‑D indexers
    val_pix = da.map_blocks(lambda blk: val_full[blk], ids_safe, dtype="float32")
    norm_pix = da.map_blocks(lambda blk: norm_full[blk], ids_safe, dtype="float32")
    raster_data = da.where(ids >= 0, _proxy.data * val_pix / norm_pix, np.nan)

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
