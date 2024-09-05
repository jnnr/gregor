from pathlib import Path

import geopandas as gpd
import rasterio as rio
import xarray as xr
from rasterstats import zonal_stats


def aggregate_raster_to_polygon(
    raster: str | Path | xr.DataArray,
    polygons: gpd.GeoSeries | gpd.GeoDataFrame,
    stats: str = "sum",
) -> gpd.GeoDataFrame:
    r"""
    Aggregate raster data to polygons.

    Parameters
    ----------
    raster : str | Path | xr.DataArray
        Path to the raster file or xarray DataArray.
    polygons : gpd.GeoSeries | gpd.GeoDataFrame
        GeoSeries or GeoDataFrame with the spatial units.
    stats : str, optional
        Statistics to compute, by default "sum".

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the original geometries
        and the aggregated statistics.
    """
    if isinstance(raster, (str, Path)):
        results_gdf = _aggregate_file_to_polygon(raster, polygons, stats)
    elif isinstance(raster, xr.DataArray):
        results_gdf = _aggregate_xarray_to_polygon(raster, polygons, stats)

    return results_gdf


def _aggregate_file_to_polygon(raster, polygons, stats, nodata=0):
    with rio.open(raster) as src:
        affine = src.transform
        array = src.read(1)

        polygons_projected = polygons.to_crs(src.crs)

        zs = zonal_stats(
            polygons_projected,
            array,
            affine=affine,
            stats=stats,
            nodata=nodata,
            geojson_out=True,
        )

        results_gdf = gpd.GeoDataFrame.from_features(zs)

        results_gdf = results_gdf.set_crs(src.crs)
        results_gdf = results_gdf.to_crs(crs=polygons.crs)

    return results_gdf


def _aggregate_xarray_to_polygon(raster, polygons, stats, nodata=0):
    # Project the polygons to the raster coordinate reference system
    polygons_projected = polygons.to_crs(raster.rio.crs)

    agg_raster_poly = zonal_stats(
        polygons_projected,
        raster.values,
        affine=raster.rio.transform(),
        stats=stats,
        nodata=nodata,
    )

    results_gdf = gpd.GeoDataFrame(
        agg_raster_poly,
        index=polygons_projected.index,
        crs=polygons_projected.crs,
        geometry=polygons_projected.geometry,
    )

    results_gdf.index.name = polygons_projected.index.name

    # Project back to the original crs
    results_gdf = results_gdf.to_crs(crs=polygons.crs)

    return results_gdf


def aggregate_point_to_polygon(
    points: gpd.GeoDataFrame, polygons: gpd.GeoSeries | gpd.GeoDataFrame, aggfunc="sum"
):
    r"""
    Aggregate point data to polygons.

    Parameters
    ----------
    points : gpd.GeoDataFrame
        GeoDataFrame containing data defined on point geometries.
    polygons : gpd.GeoSeries | gpd.GeoDataFrame
        GeoSeries or GeoDataFrame of polygon geometries.
    aggfunc : str, optional
        Aggregation function, by default "sum".

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the original geometries
        and the aggregated statistics.
    """
    if isinstance(polygons, gpd.GeoSeries):
        _polygons = polygons.to_frame()
    elif isinstance(polygons, gpd.GeoDataFrame):
        _polygons = polygons
    else:
        raise ValueError("`polygons` should be either a GeoSeries or a GeoDataFrame.")

    _polygons = _polygons[["geometry"]]

    joined_data = gpd.sjoin(points, _polygons, how="inner", predicate="within").drop(
        columns="geometry"
    )

    gpd_version = gpd.__version__
    if gpd.__version__ <= "0.14.4":
        groupby = "index_right"
        import warnings

        warnings.warn(
            f"You are using an old geopandas verion {gpd_version}. "
            "Future versions of gregor will require geopandas >= 1.0.0.",
            FutureWarning,
        )
    else:
        groupby = _polygons.index.name

    aggregated_data = joined_data.groupby(groupby).agg(aggfunc)

    result = _polygons.join(aggregated_data)

    return result
