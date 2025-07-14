# %% [markdown]
# # Guided disaggregation from polygon to raster


# import gregor
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from pathlib import Path


PATH_DATA = Path(".")
capacity_wind_ESP = gpd.read_file(PATH_DATA / "data/capacity_wind_ESP.geojson", index_col=0)
boundaries_NUTS3_ESP = gpd.read_file(PATH_DATA / "data/NUTS_3_ESP.geojson").set_index("NUTS_ID")
area_potential_ESP = rxr.open_rasterio(PATH_DATA / "data/area_potential_wind_onshore_ESP.tif").squeeze()
capacityfactor_wind_ESP  = rxr.open_rasterio(PATH_DATA / "data/capacityfactor_wind_ESP.tif").squeeze()

# %%
# boundaries_NUTS3_ESP = boundaries_NUTS3.loc[boundaries_NUTS3["CNTR_CODE"] == "ES", ["geometry"]]
# boundaries_NUTS3_ESP.to_file(PATH_DATA / "data/NUTS_3_ESP.geojson")

# #%%
# capacity_wind_ESP = capacity_wind.loc[[True if "ES" in node else False for node in capacity_wind["NUTS_ID"]]]
# capacity_wind_ESP["capacity_wind"] *= 0.1
# capacity_wind_ESP.to_file(PATH_DATA / "data/capacity_wind_ESP.geojson")

# #%%
# x = (-10, 4)  
# y = (35.6, 44)
# area_potential = area_potential.coarsen(x=10, y=10, boundary="trim").mean()
# area_potential_ESP = area_potential.rio.clip_box(minx=xlim[0], miny=ylim[0], maxx=xlim[1], maxy=ylim[1])
# area_potential_ESP.rio.to_raster(PATH_DATA / "data / area_potential_wind_onshore_ESP.tif")

#%% 
xlim = (-10, 4)
ylim = (35.6, 44)
# capacityfactor_wind = capacityfactor_wind.coarsen(x=10, y=10, boundary="trim").mean()
# capacityfactor_wind_ESP = capacityfactor_wind.rio.clip_box(minx=xlim[0], miny=ylim[0], maxx=xlim[1], maxy=ylim[1])
# capacityfactor_wind_ESP.rio.to_raster(PATH_DATA / "data/capacityfactor_wind_ESP.tif")
# plt.imshow(capacityfactor_wind.mean(dim="time").to_dataarray().squeeze())

# %% [markdown]

fig, axs = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)
capacity_wind_ESP.plot(ax=axs[0], column="capacity_wind", aspect=None)
area_potential_ESP.plot(ax=axs[1], aspect=None)
capacityfactor_wind_ESP.plot(ax=axs[2], cmap="Blues")

axs[0].set_title("Modelled wind capacity")
axs[1].set_title("Area potential")
axs[2].set_title("Mean wind capacity factor")
for ax in axs:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

#%%
import xarray as xr
import numpy as np
import rasterio as rio
from gregor.disaggregate import disaggregate_polygon_to_raster_prioritize
from gregor.aggregate import aggregate_raster_to_polygon

#%%

synth = aggregate_raster_to_polygon(area_potential_ESP, capacity_wind_ESP.geometry)
synth["capacity_wind"] = synth["sum"] * 0.4#.apply(lambda x: x * (np.random.rand() * 0.5 +0.5) , 1)
synth = synth[~synth.index.isin([30, 44, 45])]
synth

# We disaggregate to raster
wind_allocation = disaggregate_polygon_to_raster_prioritize(synth, column="capacity_wind", priority=capacityfactor_wind_ESP, limit=area_potential_ESP)
print(wind_allocation.sum())
print(synth["capacity_wind"].sum())

#%%
fig, axs = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
capacity_wind_ESP.plot(ax=axs[0], column="capacity_wind", aspect=None)
area_potential_ESP.plot(ax=axs[1], aspect=None)
capacityfactor_wind_ESP.plot(ax=axs[2], cmap="inferno", vmax=0.3)
wind_allocation.plot(ax=axs[3])
capacity_wind_ESP.geometry.boundary.plot(ax=axs[3], color="k", alpha=0.2, aspect=None)

axs[0].set_title("Modelled wind capacity")
axs[1].set_title("Area potential")
axs[2].set_title("Mean wind capacity factor")
axs[3].set_title("Allocation")

for ax in axs:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
#%%

def test_disaggregate_polygon_to_raster_guided():
    # Check that the result aggregates to the original data
    # result_agg = gregor.aggregate.aggregate_raster_to_polygon(demand_raster.FC_OTH_HH_E, boundaries_country)
    # 

    # for each region:
    # Sort the pixels by priority and compute the cumsum of the result.
    # Check that the cumsum reaches data of that region.
    # Check that pixels in order of priority are full, one partly full, then empty.
    # belongs_to = ...
    # for xx in groupby(belongs_to):
    #   yy.sort
    #   cumsum
    #   limit
    #   
    pass

