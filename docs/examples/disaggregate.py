# %% [markdown]
# # Disaggregate spatial data
# This example demonstrates how to aggregate spatial data using `gregor`.
# Imagine that you have data that is described on national level, which you want to disaggregate to a finer resolution.
# For example, you have data on energy demand per country (https://ec.europa.eu/eurostat/databrowser/view/nrg_d_hhq/default/table?lang=en), which you want to disaggregate to a raster. Assuming that
# energy demand is proportional to population density (), you can use population as a proxy to disaggregate.

# %%
import gregor
import pandas as pd
from matplotlib import pyplot as plt
import geopandas as gpd
import rioxarray as rxr
from pathlib import Path

# %%
# Load all the input data
PATH_DATA = Path(".") / "docs" / "examples"
final_energy_hh = pd.read_csv(PATH_DATA / "data/demand.csv", index_col=0)
boundaries_country = gpd.read_file(PATH_DATA / "data/boundaries_NUTS0.geojson")
boundaries_NUTS3 = gpd.read_file(PATH_DATA / "data/boundaries_NUTS3.geojson")
population = rxr.open_rasterio(PATH_DATA / "data/population_small.tif").squeeze()


# %%
final_energy_hh = boundaries_country.merge(final_energy_hh, left_on="CNTR_CODE", right_on="geo")
final_energy_hh = final_energy_hh.rename(columns={"NUTS_ID": "id"}).set_index("id")

# %%
# Plot
xlim, ylim = ((2.2, 7.5), (49, 54))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), layout="constrained")
final_energy_hh.plot(ax=ax1, column="FC_OTH_HH_E", cmap="Greens", aspect=None, legend=True)
boundaries_country.geometry.boundary.plot(ax=ax1, color="black", aspect=None)
population.rio.reproject("EPSG:4236").plot(ax=ax2, cmap="Blues", vmax=500, aspect=None)
boundaries_country.geometry.boundary.plot(ax=ax2, color="black", aspect=None)
for ax in (ax1, ax2):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
ax1.set_title("National resolution")
ax2.set_title("Population")

# %%
final_energy_hh_raster = gregor.disaggregate.disaggregate_polygon_to_raster(final_energy_hh, column="FC_OTH_HH_E", proxy=population)

# %%
xlim, ylim = ((2.5, 7.5), (49, 54))
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")

final_energy_hh.plot(ax=ax1, column="FC_OTH_HH_E", cmap="Greens", aspect=None, legend=True)
boundaries_country.geometry.boundary.plot(ax=ax1, color="black", aspect=None)

population.rio.reproject("EPSG:4236").plot(ax=ax2, cmap="Reds", vmax=500, aspect=None)
boundaries_country.geometry.boundary.plot(ax=ax2, color="black", aspect=None)

final_energy_hh_raster.rio.reproject("EPSG:4236").FC_OTH_HH_E.plot(ax=ax3, cmap="Greens", aspect=None, vmax=10)
for ax in (ax1, ax2, ax3):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

ax1.set_title("National resolution")
ax2.set_title("Population")
ax3.set_title("Disaggregated to raster")


# %%
# Compare
# gregor.aggregate.aggregate_raster_to_polygon(final_energy_hh_raster.FC_OTH_HH_E, boundaries_country)
