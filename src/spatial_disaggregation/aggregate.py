import xarray as xr
import geopandas as gpd
import rasterio as rio
from rasterstats import zonal_stats
from pathlib import Path


def aggregate_raster_to_polygon(
        raster: str|Path,
        polygons: gpd.GeoSeries|gpd.GeoDataFrame,
        stats: str="sum"
    ) -> gpd.GeoDataFrame:
    r"""
    Aggregate raster data with spatial units.
    """
    with rio.open(raster) as src:
        affine = src.transform
        array = src.read(1)
        
        polygons_projected = polygons.to_crs(src.crs)

        zs = zonal_stats(polygons_projected, array, affine=affine, stats=stats, geojson_out=True)
        
        result_gdf = gpd.GeoDataFrame.from_features(zs)

        if hasattr(raster, "name"):
            name_variable = raster.name
        else:
            name_variable = "value"

    results_gdf = result_gdf.rename(columns={stats: f"{name_variable}_{stats}"})

    return results_gdf


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
