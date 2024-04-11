"""
This example takes statistical data that is based on spatial units and
disaggregates it to raster data using a raster proxy.
"""
from spatial_aggregation import disaggregate_polygon_to_raster, aggregate_raster_to_polygon
from plot import plot_raster, plot_vector

import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
from matplotlib import pyplot as plt
import rasterio as rio


path_total_demand = "data/annual-heat-demand-twh.csv"
path_electricity = "data/annual-heat-electricity-demand-twh.csv"
path_spatial_units_national = "data/units_national.geojson"
path_spatial_units_regional = "data/units.geojson"
path_populations = "build/population-europe.tif"

# total_demand = pd.read_csv(path_total_demand)
electricity = pd.read_csv(path_electricity, index_col=[0,1,2,3])
spatial_units_national = gpd.read_file(path_spatial_units_national).set_index("id")
# spatial_units_regional = gpd.read_file(path_spatial_units_regional).set_index("id")
# populations = rioxarray.open_rasterio(path_populations) #, chunks="auto")


# load raster data
def load_raster(path):
    with rio.open(path) as file:
        raster_data = file.read()
        transform = file.transform
        crs = file.crs
        raster_data = xr.DataArray(raster_data).squeeze()
        # Add crs and transform as metadata
        raster_data.attrs['transform'] = transform
        raster_data.attrs['crs'] = crs
    return raster_data


# populations = load_raster(path_populations)

# alternative way of loading raster data
populations = rioxarray.open_rasterio(path_populations).squeeze()

# join statistical data with polygon geometries
electricity = pd.merge(electricity, spatial_units_national, left_on="country_code", right_on="id")
electricity = gpd.GeoDataFrame(electricity, geometry="geometry")
# TODO: Decide if matching of crs should happen in the disaggregate function, optionally.
electricity = electricity.to_crs(populations.rio.crs)

# disaggregate statistical data to raster
demand_electricity_rastered = disaggregate_polygon_to_raster(electricity, crs=None, proxy=populations)# proxy=populations)
print(demand_electricity_rastered)
demand_electricity_rastered.to_netcdf("build/demand_electricity_rastered.nc")

# aggregate raster data to polygons
# aggregate_raster_to_polygon()

# check
