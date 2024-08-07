from pathlib import Path

import click
import geopandas as gpd
import rioxarray as rxr

import gregor


@click.command(help="Aggregate raster data to polygon boundaries.")
@click.argument("raster", type=click.STRING)
@click.argument("polygons", type=click.STRING)
@click.argument("destination", type=click.STRING)
@click.argument("stats", type=click.STRING, default="sum")
def agg(raster, polygons, destination, stats):
    if Path(destination).exists():
        raise ValueError("Destination file already exists.")

    _raster = rxr.open_rasterio(raster).squeeze()
    _polygons = gpd.read_file(polygons)

    aggregated = gregor.aggregate.aggregate_raster_to_polygon(_raster, _polygons, stats)
    aggregated.to_file(destination)


@click.command(help="Disaggregate polygon data to raster data using proxy.")
@click.argument("data", type=click.STRING)
@click.argument("column", type=click.STRING)
@click.argument("proxy", type=click.STRING)
@click.argument("destination", type=click.STRING)
@click.option("--to-data-crs", default=False, type=click.BOOL)
def disagg(data, column, proxy, destination, to_data_crs):
    if Path(destination).exists():
        raise ValueError("Destination file already exists.")

    _data = gpd.read_file(data)
    _proxy = rxr.open_rasterio(proxy)

    # Clip proxy to extent of data for better performance
    minx, miny, maxx, maxy = _data.to_crs(_proxy.rio.crs).total_bounds
    _proxy = gregor.raster.clip(_proxy, minx, miny, maxx, maxy).squeeze()

    disaggregated = gregor.disaggregate.disaggregate_polygon_to_raster(
        _data, column, _proxy, to_data_crs
    )
    disaggregated.rio.to_raster(destination)


@click.group()
def cli():
    pass


cli.add_command(agg)
cli.add_command(disagg)
