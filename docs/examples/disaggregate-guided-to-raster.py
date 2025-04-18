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
# area_potential = area_potential.coarsen(x=100, y=100, boundary="trim").mean()
# area_potential_ESP = area_potential.rio.clip_box(minx=xlim[0], miny=ylim[0], maxx=xlim[1], maxy=ylim[1])
# area_potential_ESP.rio.to_raster(PATH_DATA / "data / area_potential_wind_onshore_ESP.tif")

#%% 
xlim = (-10, 4)
ylim = (35.6, 44)
# capacityfactor_wind = capacityfactor_wind.coarsen(x=100, y=100, boundary="trim").mean()
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
from gregor.disaggregate import get_belongs_to_matrix

#%%
def disaggregate_polygon_to_raster_prioritize(
        data_polygon: gpd.GeoDataFrame|gpd.GeoSeries,
        column: str,
        priority: xr.Dataset,
        limit: xr.Dataset,
        reproject_match: bool = True,
        to_data_crs: bool = False
    ):
    # Define crs as crs of `limit`.
    crs = limit.rio.crs
    name_result = "allocation"

    # Some checks
    if not priority.rio.crs == crs:
        raise ValueError(f"CRS mismatch! `priority` has CRS {priority.rio.crs} and `limit` {crs}.")
    
    if not len(priority.dims) == 2 and len(limit.dims) == 2:
        raise ValueError("Both `priority` and `limit` need to be 2-dimensional.")
    
    # TODO Check that aggregated limit is enough to disaggregate data

    # Transform and match data, if necessary.
    _data_polygon = data_polygon.copy()
    # compare crs. If not the same, project data to proxy's crs
    if not _data_polygon.crs == crs:
        print(
            f"CRS of `limit` ({crs}) does not match CRS of `data` ({_data_polygon.crs}). Reprojecting CRS of `data` to `proxy`'s CRS."
        )
        _data_polygon = _data_polygon.to_crs(limit.crs)

    if reproject_match:
        priority = priority.rio.reproject_match(
            limit,
            resampling=rio.enums.Resampling.sum,
            nodata=0
        ) 
    elif not priority.shape == limit.shape:
        raise ValueError(f"Shape mismatch! `priority` has shape {priority.shape} and `limit` {limit.shape}.")
    
    # Initialise an empty raster in the same shape as `limit`.
    result = xr.DataArray(coords=limit.coords, dims=limit.dims, name=name_result)
    result = result.rio.write_crs(crs)

    belongs_to = get_belongs_to_matrix(limit, _data_polygon["geometry"])

    # Loop over regions
    for id_region, data in data_polygon.iterrows():
        priority_in_region = priority.where(belongs_to == id_region)
        # Create priority_order (an ordered list of coordinates)
        priority_order = priority_in_region.stack(z=("x", "y"))
        priority_order = priority_order.sortby(lambda x: x, ascending=False)
        priority_order = priority_order.dropna("z")

        # Order limit by priority_order
        limit_in_region_ordered = limit.stack(z=("x", "y"))
        limit_in_region_ordered = limit_in_region_ordered.reindex_like(priority_order)

        # Create cumsum of limit
        cumsum = limit_in_region_ordered.cumsum()

        # Fill up pixels apart from the last one, where cumsum exceeds data[column]
        fill_up = cumsum.where(data[column] > cumsum).dropna("z")
        if not fill_up.isnull().all():
            fill_up_unstack = fill_up.unstack()
            fill_up_unstack.name = name_result

            result_to_be_overwritten = result.where(fill_up_unstack)
            assert result_to_be_overwritten.isnull().all(), "Values to be overwritten are not all NaN!"

            result = xr.merge([result, fill_up_unstack], compat="no_conflicts")

        # Find the coordinate where cumsum exceeds data[column]
        argwhere_last_pixel = np.argwhere(data[column] <= cumsum.values)[0]
        cumsum_where = cumsum[argwhere_last_pixel].unstack()
        cumsum_where.name = name_result
        rest = cumsum_where - data[column]

        result = xr.merge([result, rest], compat="no_conflicts")
        # Write rest to result at that coordinate
        # result.sel(x=coords_last_pixel["x"], y=coords_last_pixel["y"])
        # print(data[column])
        # print(fill_up)
        # print(argwhere_last_pixel)
        # print(cumsum)
        print("\n")

    return result.to_dataarray()

# We disaggregate to raster
df = capacity_wind_ESP.copy()
df["capacity_wind"] *= 0.002
wind_allocation = disaggregate_polygon_to_raster_prioritize(df, column="capacity_wind", priority=capacityfactor_wind_ESP, limit=area_potential_ESP)

#%%

fig, axs = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
capacity_wind_ESP.plot(ax=axs[0], column="capacity_wind", aspect=None)
area_potential_ESP.plot(ax=axs[1], aspect=None)
capacityfactor_wind_ESP.plot(ax=axs[2], cmap="Blues")
wind_allocation.plot(ax=axs[3])

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
print(df["capacity_wind"].sum())
print(wind_allocation.sum())
# import numpy as np
# np.where(wind_allocation.values)
# wind_allocation.values[1][8]
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

# %%
df["capacity_wind"]