# %%
import gregor
import pandas as pd
from matplotlib import pyplot as plt
import geopandas as gpd
from pathlib import Path

# %%
PATH_DATA = Path(".") / "docs" / "examples"
demand = pd.read_csv(PATH_DATA / "data/demand.csv", index_col=0)
boundaries_country = gpd.read_file(PATH_DATA / "data/boundaries_NUTS0.geojson").set_index("NUTS_ID")
boundaries_NUTS3 = gpd.read_file(PATH_DATA / "data/boundaries_NUTS3.geojson").set_index("NUTS_ID")

# %%
demand_geo = boundaries_country.join(demand)
demand_geo

# %%
uniform_proxy = gregor.disaggregate.create_uniform_proxy(demand_geo[["FC_OTH_HH_E", "geometry"]], (100, 100))
uniform_proxy

# %%

# %%