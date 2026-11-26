"""Small tests on synthetic data (run with `pytest`)."""
import os
import sys

import pytest

pytest.importorskip("geopandas")
pytest.importorskip("rasterio")

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import la_palma as lp

CRS = "EPSG:32628"


def _square_points(x0, y0, size=10.0, step=0.5):
    xs = np.arange(x0, x0 + size, step)
    return [Point(x, y) for x in xs for y in np.arange(y0, y0 + size, step)]


def test_buildings_to_polygon(tmp_path):
    pts = _square_points(0, 0) + _square_points(50, 50)
    gdf = gpd.GeoDataFrame({"x": [p.x for p in pts], "y": [p.y for p in pts]}, geometry=pts, crs=CRS)
    gdf["predicted_classification"] = 6
    out = lp.buildings_to_polygon(gdf, 1.75, CRS, tmp_path / "b.tif", tmp_path / "poly.shp", tmp_path / "pt.shp")
    assert len(out) == 2
    assert (out["area"] > lp.MIN_BUILDING_AREA).all()


def test_affected_buildings(tmp_path):
    b = gpd.GeoDataFrame({"geometry": [box(0, 0, 10, 10), box(100, 100, 110, 110)]}, crs=CRS).rename_geometry("polygons")
    b["centroids"] = b.centroid
    lava = gpd.GeoDataFrame(geometry=[box(5, 5, 20, 20)], crs=CRS)
    hit = lp.affected_buildings(b, lava, tmp_path / "a.shp", tmp_path / "ap.shp")
    assert len(hit) == 1


def test_affected_buildings_rejects_points(tmp_path):
    b = gpd.GeoDataFrame({"geometry": [box(0, 0, 10, 10)]}, crs=CRS).rename_geometry("polygons")
    b["centroids"] = b.centroid
    pts = gpd.GeoDataFrame(geometry=[Point(5, 5)], crs=CRS)
    with pytest.raises(TypeError):
        lp.affected_buildings(b, pts, tmp_path / "a.shp", tmp_path / "ap.shp")


def _write(path, data, res):
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
                       dtype="float32", crs=CRS, transform=from_origin(0, data.shape[0] * res, res, res),
                       nodata=-9999.0) as dst:
        dst.write(data.astype("float32"), 1)


def test_elevation_change(tmp_path):
    before = np.full((10, 10), 100.0)
    after = before.copy()
    after[2:5, 2:5] += 5.0
    _write(tmp_path / "before.tif", before, 2.0)
    _write(tmp_path / "after.tif", after, 2.0)
    lp.elevation_change(tmp_path / "before.tif", tmp_path / "after.tif", tmp_path / "d.tif",
                        resampling=rasterio.enums.Resampling.nearest)
    with rasterio.open(tmp_path / "d.tif") as d:
        diff = d.read(1)
    assert np.isclose(diff[3, 3], -5.0) and np.isclose(diff[8, 8], 0.0)
