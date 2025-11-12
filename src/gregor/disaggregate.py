import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.features import rasterize


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
        proxy_da = proxy
    else:  # proxy is a Dataset
        if len(proxy.data_vars) == 1:
            var_name = next(iter(proxy.data_vars))
            proxy_da = proxy[var_name]
        else:
            raise ValueError(
                f"Cannot compute multi-variable Dataset of length {len(proxy.data_vars)}. "
                "Pass a DataArray instead."
            )

    gdf = data.copy()
    index_name = gdf.index.name or "id"
    gdf.index.name = index_name

    if proxy.rio.crs != gdf.crs:
        print(
            f"CRS of `proxy` ({proxy.rio.crs}) does not match CRS of `data` ({data.crs}). Reprojecting CRS of `data` to `proxy`'s CRS."
        )
        gdf = gdf.to_crs(proxy.rio.crs)

    # always one DataArray in float32, chunked
    proxy_da = proxy_da.astype("float32").chunk({"y": chunk_size, "x": chunk_size})

    # raster of polygon IDs ─ always burn row numbers (0…n‑1)
    if not np.issubdtype(gdf.index.dtype, np.integer):
        print("Non-integer index detected, resetting to numeric for processing.")
        geom_id_source = gdf.reset_index(drop=True).geometry  # integer IDs
    else:
        geom_id_source = gdf.geometry

    belongs_to = get_belongs_to_matrix(proxy_da, geom_id_source)
    belongs_to = belongs_to.where(~belongs_to.isnull(), other=-1)
    belongs_to = belongs_to.astype("int32").chunk(proxy_da.chunks)

    # zonal sums per polygon
    max_id = int(belongs_to.max().compute())  # get the 'true' maximum
    n_ids = max_id + 1  # valid IDs: 0 … max_id

    def _zonal_sum(block_proxy, block_ids, *, n_ids):
        ok = (block_ids >= 0) & np.isfinite(block_proxy)  # NEW – drop NaNs
        return np.bincount(
            block_ids[ok].ravel(),
            weights=block_proxy[ok].ravel(),
            minlength=n_ids,
        ).astype("float32")

    zonal = xr.apply_ufunc(
        _zonal_sum,
        proxy_da,
        belongs_to,
        kwargs={"n_ids": n_ids},
        input_core_dims=[["y", "x"], ["y", "x"]],
        output_core_dims=[[index_name]],
        output_sizes={index_name: n_ids},
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32"],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    # values and normalisations as dense vectors
    normalisation = zonal.sum(dim=set(zonal.dims) - {index_name}).compute()

    # +1 extra slot (sentinel) that will later propagate NaN outside given geometries
    val_full = np.zeros(n_ids + 1, dtype="float32")
    norm_full = np.zeros(n_ids + 1, dtype="float32")
    sentinel = n_ids  # index of zero‑filled slot

    val_full[sentinel] = np.nan
    norm_full[sentinel] = 1.0  # avoid 0/0 inside map_blocks

    val_full[: len(gdf)] = gdf[column].astype("float32").values
    norm_full[:n_ids] = normalisation.values

    # raster assembly
    ids = belongs_to.data  # (y,x)
    ids_safe = da.where(ids >= 0, ids, sentinel).astype(
        "int32"
    )  # max > billion polygons

    # lookup via map_blocks, works with N‑D indexers
    val_pix = da.map_blocks(lambda blk: val_full[blk], ids_safe, dtype="float32")
    norm_pix = da.map_blocks(lambda blk: norm_full[blk], ids_safe, dtype="float32")
    raster_data = da.where(ids >= 0, proxy_da.data * val_pix / norm_pix, np.nan)

    raster = xr.DataArray(
        raster_data,
        dims=proxy_da.dims,
        coords=proxy_da.coords,
        name=column,
        attrs=proxy_da.attrs,
    )

    if to_data_crs and proxy_da.rio.crs != data.crs:
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
