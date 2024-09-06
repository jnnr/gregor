# This script is an example of how to use the command line interface of Gregor.

gregor poly-raster docs/examples/data/demand.geojson FC_OTH_HH_E docs/examples/data/population_small.tif poly-raster.tif

gregor poly-point docs/examples/data/demand.geojson FC_OTH_HH_E docs/examples/data/cities.geojson pop_max poly-point.geojson

gregor raster-poly poly-raster.tif docs/examples/data/boundaries_NUTS3.geojson raster-poly.geojson

gregor point-poly poly-point.geojson docs/examples/data/boundaries_NUTS3.geojson point-poly.geojson
