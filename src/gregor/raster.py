from pathlib import Path

import geopandas as gpd
import rasterio as rio
import xarray as xr
from shapely.geometry import box


# TODO read the docs first
# https://corteva.github.io/rioxarray/html/examples/clip_geom.html#Clipping-larger-rasters
# can use from_disk=True


def clip(raster: str | Path | xr.DataArray, minx, miny, maxx, maxy, destination=None):
    r"""
    Clip raster to bounding box.

    Parameters
    ----------
    raster : str | Path | xr.DataArray
        Path to the raster file or xarray DataArray.
    minx : float
        Minimum x-coordinate of the bounding box.
    miny : float
        Minimum y-coordinate of the bounding box.
    maxx : float
        Maximum x-coordinate of the bounding box.
    maxy : float
        Maximum y-coordinate of the bounding box.
    destination : str | Path, optional
        Path to save the clipped raster, by default None.

    Returns
    -------
    xr.DataArray
        Clipped raster.
    """
    if isinstance(raster, xr.DataArray):
        return _clip_xarray(raster, minx, miny, maxx, maxy)
    if isinstance(raster, str) or isinstance(raster, Path):
        if destination is None:
            raise ValueError("Destination must be provided when clipping a file.")
        _clip_file(raster, minx, miny, maxx, maxy, destination)


def _clip_xarray(raster, minx, miny, maxx, maxy):
    return raster.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)


def _clip_file(raster, minx, miny, maxx, maxy, destination):
    bbox = box(minx, miny, maxx, maxy)
    bbox = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs="EPSG:4326")

    with rio.open(raster) as src:
        # Reproject the bounding box to the raster's CRS
        bbox = bbox.to_crs(src.crs)

        # Clip raster
        out_raster, out_transform = rio.mask.mask(src, bbox.geometry, crop=True)

        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_raster.shape[1],
                "width": out_raster.shape[2],
                "transform": out_transform,
            }
        )

        # Save the clipped raster to a new file
        with rio.open(destination, "w", **out_meta) as dest:
            dest.write(out_raster)
