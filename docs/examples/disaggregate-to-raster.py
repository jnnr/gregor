# %% [markdown]
# # Polygon to raster (and back)
# This example demonstrates how to aggregate spatial data using `gregor`.
# Imagine that you have some data that is described on national level, which you want to disaggregate to a finer resolution.
# This could be household energy demand per country, which is provided by EUROSTAT (https://ec.europa.eu/eurostat/databrowser/view/nrg_d_hhq/default/table?lang=en).
# Ideally, you would have another source with higher resolution, but in lack of that, you want to use some assumptions to disaggregate your data to higher resolution.
# Assuming that energy demand is proportional to population density, you want use population data () as a proxy.
# `gregor` helps you doing that.
#
# First, import the necessary packages and data on household energy demand, boundaries on country and NUTS3 resolution and population data.

# %%
import gregor
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
import rioxarray as rxr
from pathlib import Path

# %%
# Load all the input data
PATH_DATA = Path(".") / "docs" / "examples"
demand = pd.read_csv(PATH_DATA / "data/demand.csv", index_col=0)
boundaries_country = gpd.read_file(PATH_DATA / "data/boundaries_NUTS0.geojson").set_index("NUTS_ID")
boundaries_NUTS3 = gpd.read_file(PATH_DATA / "data/boundaries_NUTS3.geojson").set_index("NUTS_ID")
population = rxr.open_rasterio(PATH_DATA / "data/population_small.tif").squeeze()

# %% [markdown]
# Here, we merge the demand data with the boundaries on country level, to connect the energy demand with the geometries.

# %%
demand_geo = boundaries_country.join(demand)
demand_geo

# %% [markdown]
# This is how our inital data looks like.

# %%
# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 4), layout="constrained")

xlim, ylim = ((2.2, 7.5), (49, 54))
reds = LinearSegmentedColormap.from_list('reds', ['white', 'red'])
greens = LinearSegmentedColormap.from_list('greens', ['white', 'Green'])

demand_geo.plot(ax=ax1, column="FC_OTH_HH_E", cmap=greens, aspect=None, legend=True, legend_kwds={'location': 'bottom', 'label': "GWh/year"})
boundaries_country.geometry.boundary.plot(ax=ax1, color="black", aspect=None)
population.rio.reproject("EPSG:4236").plot.imshow(ax=ax2, cmap=reds, vmax=500, aspect=None, add_colorbar=True, cbar_kwargs={'location': 'bottom', 'label': "# inhabitants"})
boundaries_country.geometry.boundary.plot(ax=ax2, color="black", aspect=None)
for ax in (ax1, ax2):
    ax.set_xlim(*xlim)
    ax.set_axis_off()
    ax.set_ylim(*ylim)

ax1.set_title("National resolution")
ax2.set_title("Population")
plt.show()

# %% [markdown]
# Now, we disaggregate the demand data using the population data as a proxy. The result is a raster dataset with the resolution of the proxy.

# %%
demand_raster = gregor.disaggregate.disaggregate_polygon_to_raster(demand_geo, column="FC_OTH_HH_E", proxy=population)

# %% [markdown]
# Aggregate the raster data back to countries for checking. The result should be equal (up to numerics) to the original data.
# %%
demand_aggregated = gregor.aggregate.aggregate_raster_to_polygon(demand_raster.FC_OTH_HH_E, boundaries_country)

# %%
# Compare with original demand
comparison = pd.DataFrame(index=demand_aggregated.index)
comparison["original"] = demand_geo["FC_OTH_HH_E"]
comparison["aggregated"] = demand_aggregated["sum"]
comparison["relative diff"] = abs((comparison["original"] - comparison["aggregated"])/ comparison["original"])
assert (comparison["relative diff"] < 1e-6).all()
comparison

# %% [markdown]
# Finally, we aggregate the raster data to NUTS3 level, which is the resolution we are interested in.

# %%
demand_NUTS3 = gregor.aggregate.aggregate_raster_to_polygon(demand_raster.FC_OTH_HH_E, boundaries_NUTS3)

# %% [markdown]
# This is a plot of the original data and the disaggregated data in raster format, as well as the data aggregate to NUTS3 resolution.
# %%

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(9, 4), layout="constrained")

xlim, ylim = ((2.5, 7.5), (49, 54))
vmax = demand_NUTS3["sum"].max()

demand_geo.plot(ax=ax1, column="FC_OTH_HH_E", vmin=0, vmax=vmax, cmap=greens, aspect=None, legend=True, legend_kwds={'location': 'bottom', 'label': "GWh/year"})
boundaries_country.geometry.boundary.plot(ax=ax1, color="black", linewidth=1, aspect=None)

population.rio.reproject("EPSG:4236").plot.imshow(ax=ax2, cmap=reds, vmax=500, aspect=None, add_colorbar=True, cbar_kwargs={'location': 'bottom', 'label': "# inhabitants"})
boundaries_country.geometry.boundary.plot(ax=ax2, color="black", linewidth=1, aspect=None)

demand_raster.rio.reproject("EPSG:4236").FC_OTH_HH_E.plot.imshow(ax=ax3, cmap=greens, aspect=None, vmax=10, add_colorbar=True, cbar_kwargs={'location': 'bottom', 'label': "GWh/year"})
boundaries_country.geometry.boundary.plot(ax=ax3, color="black", linewidth=1, aspect=None)

demand_NUTS3.plot(ax=ax4, column="sum", vmin=0, vmax=vmax, cmap=greens, aspect=None, legend=True, legend_kwds={'location': 'bottom', 'label': "GWh/year"})
demand_NUTS3.geometry.boundary.plot(ax=ax4, color="black", linewidth=1, aspect=None)

for ax in (ax1, ax2, ax3, ax4):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axis_off()

ax1.set_title("National\nresolution")
ax2.set_title("Proxy\n(population)")
ax3.set_title("Disaggregated\nto raster")
ax4.set_title("Aggregated\nto NUTS3")

plt.show()