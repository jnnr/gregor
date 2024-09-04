import geopandas as gpd
import numpy as np
import pandas as pd
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
    index_name = _data.index.name
    if index_name is None:
        index_name = "id"
        _data.index.name = index_name

    if not proxy.rio.crs == data.crs:
        print(
            f"CRS of `proxy` ({proxy.rio.crs}) does not match CRS of `data` ({data.crs}). Reprojecting CRS of `data` to `proxy`'s CRS."
        )
        _data = _data.to_crs(proxy.rio.crs)

    # Each raster point belongs to one spatial_unit
    belongs_to = get_belongs_to_matrix(proxy, _data.geometry)
    _data = _data[[column]].to_xarray()
    normalization = proxy.groupby(belongs_to).sum().rename(group=index_name)

    # # Remove regions that do not belong to any geometry
    _data = _data.sel({index_name: normalization.coords[index_name]})

    # Disaggregate data to raster using proxy
    # raster_{x,y} = 1/normalization_{id} * _data_{id} * belongs_to_{id,x,y} * proxy_{x,y}
    raster = xr.DataArray(data=0, dims=["y", "x"], coords={"y": proxy.y, "x": proxy.x})
    for id in normalization.coords[index_name]:
        raster_id = (
            1
            / normalization.sel({index_name: id})
            * _data.sel({index_name: id})
            * (belongs_to == id)
            * proxy
        )
        raster = raster + raster_id

    if to_data_crs:
        print(f"Reprojecting results to `data`'s CRS {data.crs}.")
        raster = raster.rio.reproject(data.crs)

    return raster


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


def get_belongs_to_matrix(raster: xr.Dataset, polygons: gpd.GeoSeries) -> xr.Dataset:
    r"""
    Get a matrix which indicates which polygon each raster point belongs to.

    Parameters
    ----------
    raster : xr.Dataset
        Raster data to get the matrix for.
    polygons : gpd.GeoSeries
        Polygons to compute the matrix for.

    Returns
    -------
    xr.Dataset
        Matrix which indicates which polygon each raster point belongs to.
    """
    assert len(raster.dims) == 2, "Raster data should have 2 dimensions."
    # create an empty dataarray with the coords matching raster and spatial_units
    belongs_to_matrix = xr.DataArray(
        data=None, dims=["y", "x"], coords={"y": raster.y, "x": raster.x}
    )
    belongs_to_matrix.attrs["transform"] = raster.rio.transform
    belongs_to_matrix.attrs["crs"] = raster.rio.crs

    for id, geometry in polygons.items():
        mask = geometry_mask(
            [geometry],
            out_shape=raster.shape,
            transform=raster.rio.transform(),
            invert=True,
        )
        mask = xr.DataArray(mask, coords=raster.coords, dims=raster.dims)
        # assert belongs_to_matrix.where(mask).isnull().all(), "Trying to assign to value which is not None. Maybe cause of overlapping geometries."
        belongs_to_matrix = belongs_to_matrix.where(~mask, id)

    return belongs_to_matrix


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
    assert (
        points.belongs_to.apply(len).max() == 1
    ), "Every Point should belongs to exaxtly one polygon."
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
