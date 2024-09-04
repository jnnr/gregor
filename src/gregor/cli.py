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
def raster_poly(raster, polygons, destination, stats):
    if Path(destination).exists():
        raise ValueError("Destination file already exists.")

    _raster = rxr.open_rasterio(raster).squeeze()
    _polygons = gpd.read_file(polygons)

    aggregated = gregor.aggregate.aggregate_raster_to_polygon(_raster, _polygons, stats)
    aggregated.to_file(destination)


@click.command(help="Aggregate point data to polygon boundaries.")
@click.argument("point", type=click.STRING)
@click.argument("polygons", type=click.STRING)
@click.argument("destination", type=click.STRING)
@click.argument("stats", type=click.STRING, default="sum")
def point_poly(point, polygons, destination, stats):
    if Path(destination).exists():
        raise ValueError("Destination file already exists.")

    _point = gpd.read_file(point)
    _polygons = gpd.read_file(polygons)

    aggregated = gregor.aggregate.aggregate_point_to_polygon(_point, _polygons, stats)
    aggregated.to_file(destination)


@click.command(help="Disaggregate polygon data using raster proxy.")
@click.argument("data", type=click.STRING)
@click.argument("column", type=click.STRING)
@click.argument("proxy", type=click.STRING)
@click.argument("destination", type=click.STRING)
@click.option("--to-data-crs", default=False, type=click.BOOL)
def poly_raster(data, column, proxy, destination, to_data_crs):
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


@click.command(help="Disaggregate polygon data using point proxy.")
@click.argument("data", type=click.STRING)
@click.argument("column", type=click.STRING)
@click.argument("proxy", type=click.STRING)
@click.argument("proxy_column", type=click.STRING)
@click.argument("destination", type=click.STRING)
@click.option("--to-data-crs", default=False, type=click.BOOL)
def poly_point(data, column, proxy, proxy_column, destination, to_data_crs):
    if Path(destination).exists():
        raise ValueError("Destination file already exists.")

    _data = gpd.read_file(data)
    _proxy = gpd.read_file(proxy)

    disaggregated = gregor.disaggregate.disaggregate_polygon_to_point(
        _data, column, _proxy, proxy_column, to_data_crs
    )
    disaggregated.to_file(destination)


@click.group()
def cli():
    pass


cli.add_command(raster_poly)
cli.add_command(point_poly)
cli.add_command(poly_raster)
cli.add_command(poly_point)
