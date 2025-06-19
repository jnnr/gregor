import pandas as pd
import geopandas as gpd
import xarray as xr
from gregor.aggregate import _aggregate_xarray_to_polygon
from gregor.disaggregate import disaggregate_polygon_to_raster


def aggregate_timeseries_raster_to_polygon(
    raster: xr.DataArray,
    polygons: gpd.GeoSeries | gpd.GeoDataFrame,
    stats: str = "sum",
) -> pd.DataFrame:
    r"""
    Aggregate raster data to polygons.

    Parameters
    ----------
    raster : xr.DataArray
        Path to the raster file or xarray DataArray.
    polygons : gpd.GeoSeries | gpd.GeoDataFrame
        GeoSeries or GeoDataFrame with the spatial units.
    stats : str, optional
        Statistics to compute, by default "sum".

    Returns
    -------
    pd.DataFrame
        DataFrame with the aggregated timeseries per polygon.
    """
    results = {}
    for t in raster.time.values:
        raster_t = raster.sel(time=t)
        results_gdf_t = _aggregate_xarray_to_polygon(raster_t, polygons, stats)
        results_gdf_t = pd.DataFrame(results_gdf_t.drop(columns="geometry"))
        results[t] = results_gdf_t[stats]

    # Combine results into a DataFrame
    results_df = pd.DataFrame(
        results,
        index=polygons.index,
    ).T

    return results_df


def disaggregate_timeseries_polygon_to_raster(
    data: pd.DataFrame,
    geometries: gpd.GeoDataFrame,
    column: str,
    proxy: xr.Dataset,
    to_data_crs: bool = False,
) -> xr.Dataset:
    # iteratively apply disaggregate_polygon_to_raster
    # to each time step of the raster data
    rasters = []
    for t in data.index:
        data_t = data.loc[t]
        data_t = gpd.GeoDataFrame(data_t, geometry=geometries.geometry, crs=geometries.crs)
        data_t = data_t.rename(columns={t: column})
 
        raster_t = disaggregate_polygon_to_raster(
            data_t,
            column=column,
            proxy=proxy,
            to_data_crs=to_data_crs,
        )
        rasters.append(raster_t)

    # Combine results into a Dataset
    rasters_ds = xr.concat(rasters, dim="time")
    rasters_ds = rasters_ds.assign_coords(time=data.index)
    rasters_ds.rio.write_crs(rasters[0].rio.crs)
    return rasters_ds
