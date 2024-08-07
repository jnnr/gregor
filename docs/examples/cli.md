It is also possible to run Gregor via the command line interface, without writing any python code. With Gregor installed in your python environment, you 
can for example disaggregate the example data:

    gregor disagg docs/examples/data/demand.geojson FC_OTH_HH_E docs/examples/data/population_small.tif disaggregated.tif

And then aggregate the disaggregated raster data.

    gregor agg disaggregated.tif docs/examples/data/boundaries_NUTS3.geojson aggregated.geojson
