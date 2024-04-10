"""
This example takes 
Statistical data on units and disaggregates them to raster using a proxy.


total_demand = "build/data/heat/annual-heat-demand-twh.csv",
electricity = "build/data/heat/annual-heat-electricity-demand-twh.csv",

Replace the following with polygons and raster data:
locations = "build/data/regional/units.csv", just get the geojson
populations = "build/data/regional/population.csv", population data is processed in electricity autarky paper
"""
from spatial_aggregation import disaggregate_polygon_to_raster, aggregate_raster_to_polygon
from plot import plot_raster, plot_vector

import ipdb
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
from matplotlib import pyplot as plt

path_total_demand = "data/annual-heat-demand-twh.csv"
path_electricity = "data/annual-heat-electricity-demand-twh.csv"
path_spatial_units_national = "data/units_national.geojson"
path_spatial_units_regional = "data/units.geojson"
path_populations = "build/population-europe.tif"

# total_demand = pd.read_csv(path_total_demand)
electricity = pd.read_csv(path_electricity, index_col=[0,1,2,3])
spatial_units_national = gpd.read_file(path_spatial_units_national).set_index("id")
# spatial_units_regional = gpd.read_file(path_spatial_units_regional).set_index("id")
populations = rioxarray.open_rasterio(path_populations) #, chunks="auto")
# populations = populations.coarsen(x=100, y=100, boundary="trim").mean()
populations = populations.rio.write_crs("EPSG:4326")

# print(populations)
# print(total_demand)
# print(electricity)
# print(locations)
# print(populations.rio.crs)

# join statistical data with polygon geometries
electricity = pd.merge(electricity, spatial_units_national, left_on="country_code", right_on="id")
electricity = gpd.GeoDataFrame(electricity, geometry="geometry")
# breakpoint()

# plot statistical data and proxies
# plot_vector(electricity, column="value", cmap="Reds", legend=True)
# plt.savefig("build/electricity.png")

# plot_raster(populations.squeeze(), cmap="Blues", vmin=0, vmax=100)
# plt.savefig("build/population.png")

# disaggregate statistical data to raster
demand_electricity_rastered = disaggregate_polygon_to_raster(electricity, crs=None, resolution=(10,10))# proxy=populations)
print(demand_electricity_rastered)
demand_electricity_rastered.to_netcdf("build/demand_electricity_rastered.nc")

# aggregate raster data to polygons
# aggregate_raster_to_polygon()

# check
