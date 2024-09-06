It is also possible to run Gregor via the command line interface, without writing any python code. With Gregor installed in your python environment, you 
can disaggregate the example polygon data on energy demand to a raster, assuming population as a proxy.

``` bash
gregor poly-raster docs/examples/data/demand.geojson FC_OTH_HH_E docs/examples/data/population_small.tif poly-raster.tif
```

You can aggregate the raster data back to polygons.

``` bash
gregor raster-poly poly-raster.tif docs/examples/data/boundaries_NUTS3.geojson raster-poly.geojson
```

Alternatively, you can disaggregate the same polygon data to a points, working with city population as a proxy.

``` bash
gregor poly-point docs/examples/data/demand.geojson FC_OTH_HH_E docs/examples/data/cities.geojson pop_max poly-point.geojson
```

And, finally, you can also aggregate the point data back to polygons.

``` bash
gregor point-poly poly-point.geojson docs/examples/data/boundaries_NUTS3.geojson point-poly.geojson
```
