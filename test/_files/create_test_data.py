import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from shapely.geometry import Point, Polygon


def get_random_point(xlim: tuple, ylim: tuple):
    return Point(
        np.random.rand() * (xlim[1] - xlim[0]) + xlim[0],
        np.random.rand() * (ylim[1] - ylim[0]) + ylim[0],
    )


def get_square_segmentation(xlim, ylim, resolution):
    size = (xlim[1] - xlim[0]) / resolution
    polygons = []
    for i in range(resolution):
        for j in range(resolution):
            x, y = xlim[0] + i * size, ylim[0] + j * size
            polygons.append(
                Polygon([(x, y), (x + size, y), (x + size, y + size), (x, y + size)])
            )
    return polygons


def get_polygon_segmentation(xlim, ylim):
    polygons = []
    polygons.append(
        Polygon(
            [
                (xlim[0], ylim[0]),
                (xlim[1], ylim[0]),
                (xlim[1], (ylim[0] + ylim[1]) / 2),
                (xlim[0], ylim[1]),
            ]
        )
    )
    polygons.append(
        Polygon(
            [
                (xlim[0], ylim[0]),
                (xlim[1], (ylim[0] + ylim[1]) / 2),
                (xlim[0], ylim[1]),
            ]
        )
    )
    polygons = gpd.GeoSeries(polygons, crs="EPSG:4326")
    return polygons

def get_index(resolution):
    index = np.arange(resolution * resolution)
    index = index.reshape((resolution, resolution))
    index = np.rot90(index)
    index = index.flatten().tolist()
    index.reverse()
    
    return index

if __name__ == "__main__":
    # Raster needs to be created north up.
    # https://github.com/perrygeo/python-rasterstats/issues/218#issuecomment-640377751
    resolution = 4
    dummy_raster = xr.DataArray(
        data=np.random.choice([0, 0.25, 0.5, 0.75, 1.0], size=(resolution, resolution)),
        coords={
            "lat": np.linspace(11.5, 10, resolution),
            "lon": np.linspace(0, 1.5, resolution),
        },
        dims=["lat", "lon"],
    )
    dummy_raster = dummy_raster.rio.write_crs("EPSG:4326")
    dummy_raster = dummy_raster.rio.set_spatial_dims("lon", "lat")
    dummy_raster.rio.write_crs("EPSG:4326").rio.to_raster("raster.tif")

    segmentation_2 = gpd.GeoDataFrame(
        {
            "id": get_index(2),
            "geometry": get_square_segmentation(
                (-0.25, 1.75), (9.75, 11.75), 2
            ),
        },
        crs="EPSG:4326",
    ).set_index("id").sort_index()
    segmentation_2.to_file("segmentation_2x2.geojson", driver="GeoJSON")


    segmentation_3 = gpd.GeoDataFrame(
        {
            "id": get_index(3),
            "geometry": get_square_segmentation(
                (-0.0, 1.5), (10, 11.5), 3
            ),
        },
        crs="EPSG:4326",
    ).set_index("id").sort_index()
    segmentation_3.to_file("segmentation_3x3.geojson", driver="GeoJSON")

    segmentation_polygon = get_polygon_segmentation((-0.25, 1.75), (9.75, 11.75))
    segmentation_polygon.to_file("segmentation_polygon.geojson", driver="GeoJSON")

    n_points = 10
    dummy_points = gpd.GeoDataFrame(
        {
            "id": range(n_points),
            "data": np.random.rand(n_points),
            "geometry": [get_random_point((0, 1), (10, 11)) for _ in range(n_points)],
        },
        crs="EPSG:4326",
    ).set_index("id")
    dummy_points.to_file("points.geojson", driver="GeoJSON")

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    dummy_raster.plot(ax=ax, cmap="Greens")
    segmentation_2.boundary.plot(ax=ax, color="red", label="2x2 segmentation")
    segmentation_3.boundary.plot(ax=ax, color="blue", label="3x3 segmentation")
    segmentation_polygon.boundary.plot(ax=ax, facecolor="none", edgecolor="black", label="Polygon segmentation")
    dummy_points.plot(ax=ax, color="red", label="Point proxy")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=2,
        fontsize="small",
    )
    plt.savefig("test.png", bbox_inches="tight")
