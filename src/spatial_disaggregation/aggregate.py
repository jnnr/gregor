import xarray as xr
import geopandas as gpd
from rasterstats import zonal_stats


def aggregate_raster_to_polygon(
        raster_data: xr.Dataset|str,
        polygons: gpd.GeoSeries|gpd.GeoDataFrame,
        transform: None
) -> gpd.GeoDataFrame:
    r"""
    Aggregate raster data with spatial units and aggregate the data.
    """
    # Aggregate raster data to spatial units
    if isinstance(polygons, gpd.GeoSeries):
        aggregated_data = polygons.to_frame()
    elif isinstance(polygons, gpd.GeoDataFrame):
        aggregated_data = polygons.copy()
    else:
        raise ValueError("`polygons` should be either a GeoSeries or a GeoDataFrame.")
    if isinstance(raster_data, str):
        raise NotImplementedError("Not implemented yet.")

    if hasattr(raster_data, "name"):
        name_variable = raster_data.name
    else:
        name_variable = "value"
    print(aggregated_data)
    a = zonal_stats(polygons.geometry, raster_data) #, affine=transform, stats=['mean', 'sum', 'count'])
    print(a)
    aggregated_data[name_variable] = a
    return aggregated_data


def aggregate_point_to_polygon(points: gpd.GeoDataFrame, polygons: gpd.GeoSeries|gpd.GeoDataFrame, aggfunc='sum'):
    if isinstance(polygons, gpd.GeoSeries):
        _polygons = polygons.to_frame()
    elif isinstance(polygons, gpd.GeoDataFrame):
        _polygons = polygons
    else:
        raise ValueError("`polygons` should be either a GeoSeries or a GeoDataFrame.")

    joined_data = gpd.sjoin(points, _polygons, how="inner", op="within").drop(columns="geometry")

    aggregated_data = joined_data.groupby("index_right").agg(aggfunc)

    result = _polygons.join(aggregated_data)

    return result
