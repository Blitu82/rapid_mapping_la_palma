"""
Rapid mapping of the 2021 La Palma (Cumbre Vieja) eruption.

Pipeline:
  1. Load an airborne lidar point cloud (LAS) and classify building points (Random Forest).
  2. Convert building points to building polygons / centroids (rasterise + vectorise).
  3. Cluster a Sentinel image (K-Means), pick the lava cluster, and extract the lava flow
     polygon (DBSCAN + alpha shape).
  4. Intersect buildings with the lava flow to find affected buildings.
  5. Compute elevation change between a pre- and a post-eruption DTM/DSM.
  6. Optionally load the results into a PostGIS database.

All paths are relative to --data-dir / --output-dir (see `build_paths`); nothing is
hard-coded to a particular machine. Database credentials come from environment
variables (or a .env file) and the password is asked for interactively if missing.

Usage:
    python la_palma.py --data-dir data --output-dir output
    python la_palma.py --lava-cluster 3 --no-plots --to-db
"""
import argparse
import getpass
import math
import os
import pickle
import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
from rasterio.transform import xy as transform_xy
from shapely.geometry import MultiPoint, shape
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DESIRED_CRS = CRS.from_epsg(32628)  # WGS 84 / UTM 28N
RESOLUTION = 1.75                   # raster cell size (m) used to vectorise buildings
MIN_BUILDING_AREA = 10.0            # m2, smaller polygons are discarded
BUILDING_CLASS = 6                  # ASPRS lidar class "Building"


# --------------------------------------------------------------------------- paths
def build_paths(data_dir, output_dir):
    """Return a dict with every input and output path used by the pipeline."""
    d, o = Path(data_dir), Path(output_dir)
    return {
        # inputs
        "aoi_las": d / "las" / "lidar_AOI.las",
        "image": d / "sentinel" / "sentinel_lava.tif",
        "dtm_before": d / "dtm" / "before" / "Before_MDT02_REGCAN95_UTM28N_2m.tiff",
        "dtm_after": d / "dtm" / "after" / "Cumbre_Vieja_DSM_SfM_January_2022_20cm_REGCAN95.tif",
        # outputs
        "rf_model": o / "random_forest_classifier" / "RF_model.pickle",
        "buildings_raster": o / "output_raster" / "buildings.tif",
        "lava_raster": o / "output_raster" / "classified_lava.tif",
        "buildings_polygon": o / "output_shape" / "buildings_rasterize" / "buildings_polygon.shp",
        "buildings_point": o / "output_shape" / "buildings_rasterize" / "buildings_points.shp",
        "buildings_dbscan": o / "output_shape" / "buildings_dbscan" / "buildings_dbscan.shp",
        "lava_flow": o / "output_shape" / "lava_flow" / "lava_flow.shp",
        "affected_polygon": o / "output_shape" / "affected_buildings" / "affected_buildings_area.shp",
        "affected_point": o / "output_shape" / "affected_buildings" / "affected_buildings_point.shp",
        "difference": o / "DTM_difference" / "difference.tif",
    }


def _ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _finish_plot(show):
    if show:
        plt.show()
    else:
        plt.close()


# ------------------------------------------------------------------- lidar -> buildings
def import_las_to_geoDataFrame(in_file, desired_crs):
    """
    Read a LAS file into a GeoDataFrame with columns x, y, z, classification, geometry.
    """
    print("Loading LAS file...")
    las = laspy.read(in_file)
    x, y, z = np.array(las.x), np.array(las.y), np.array(las.z)
    df = pd.DataFrame({"x": x, "y": y, "z": z, "classification": np.array(las.classification)})
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x, y), crs=desired_crs)
    print(f"LAS file loaded: {len(gdf):,} points.")
    return gdf


def random_forest_classifier(lidar_gdf, model_output_path, random_state=100, n_jobs=-1):
    """
    Train a Random Forest on x, y, z to predict the lidar classification, save it as a
    pickle file and return a copy of the input with a 'predicted_classification' column.

    NOTE: the features are the point coordinates only and the hold-out set is a random
    20 % of the same point cloud, so the reported accuracy is optimistic for unseen areas.
    """
    X = lidar_gdf[["x", "y", "z"]]
    y = lidar_gdf["classification"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    rf = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
    print("Fitting Random Forest classifier...")
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(f"Random Forest accuracy (random hold-out): {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    _ensure_parent(model_output_path)
    with open(model_output_path, "wb") as fh:
        pickle.dump(rf, fh)
    print(f"Random Forest model saved: {model_output_path}")

    out = lidar_gdf.copy()
    out["predicted_classification"] = rf.predict(X)
    return out


def buildings_to_polygon(lidar_classified_gdf, resolution, desired_crs, output_raster,
                         output_shape_buildings, output_shape_buildings_points,
                         min_area=MIN_BUILDING_AREA):
    """
    Rasterise predicted building points, vectorise the raster into polygons, drop polygons
    smaller than `min_area` m2 and save polygons and centroids as shapefiles.

    Returns a GeoDataFrame whose active geometry column is 'polygons' and which also has a
    'centroids' geometry column and an 'area' column (m2).
    """
    pts = lidar_classified_gdf[lidar_classified_gdf["predicted_classification"] == BUILDING_CLASS]
    if pts.empty:
        raise ValueError("No points predicted as buildings (class 6).")
    minx, miny, maxx, maxy = pts.total_bounds
    cols = max(1, math.ceil((maxx - minx) / resolution))
    rows = max(1, math.ceil((maxy - miny) / resolution))
    transform = from_origin(minx, maxy, resolution, resolution)

    raster = rasterize(shapes=pts.geometry, out_shape=(rows, cols), transform=transform,
                       fill=0, default_value=255, dtype=np.uint8)

    _ensure_parent(output_raster)
    profile = {"driver": "GTiff", "dtype": "uint8", "count": 1, "width": cols, "height": rows,
               "transform": transform, "crs": desired_crs}
    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(raster, 1)
    print(f"Building raster saved: {output_raster}")

    polygons = [shape(geom) for geom, value in shapes(raster, mask=raster == 255, transform=transform)
                if value == 255]
    gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=desired_crs).rename_geometry("polygons")
    gdf["centroids"] = gdf.centroid
    gdf["area"] = gdf.area
    filtered = gdf[gdf["area"] > min_area].copy()

    _ensure_parent(output_shape_buildings)
    filtered["polygons"].to_file(output_shape_buildings, driver="ESRI Shapefile")
    filtered["centroids"].to_file(output_shape_buildings_points, driver="ESRI Shapefile")
    print(f"{len(filtered)} buildings saved: {output_shape_buildings}")
    return filtered


def buildings_to_polygon_dbscan(lidar_classified_gdf, desired_crs, out_shapefile_buildings_dbscan,
                                eps=0.1, min_samples=5):
    """
    Alternative building extraction: DBSCAN on the (standardised) x, y of building points,
    one convex hull per cluster. `eps` refers to STANDARDISED coordinates, so it depends
    on the extent of the area of interest.
    """
    pts = lidar_classified_gdf[lidar_classified_gdf["predicted_classification"] == BUILDING_CLASS]
    xy = pts[["x", "y"]].copy()
    scaled = StandardScaler().fit_transform(xy.values)
    xy["cluster"] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
    xy = xy[xy["cluster"] != -1]

    labels, hulls = [], []
    for label, group in xy.groupby("cluster"):
        labels.append(label)
        hulls.append(MultiPoint(group[["x", "y"]].values).convex_hull)
    gdf = gpd.GeoDataFrame({"cluster": labels}, geometry=hulls, crs=desired_crs)

    _ensure_parent(out_shapefile_buildings_dbscan)
    gdf.to_file(out_shapefile_buildings_dbscan, driver="ESRI Shapefile")
    print(f"DBSCAN buildings saved: {out_shapefile_buildings_dbscan}")
    return gdf


# ----------------------------------------------------------------- Sentinel -> lava flow
def lava_cluster_plotter(in_image, out_raster_lava_classification, n_clusters=10,
                         random_state=42, show=True):
    """
    K-Means on band 1 of a satellite image. Saves the cluster image as GeoTIFF and returns
    a GeoDataFrame with one point per pixel (pixel centre) and a 'cluster_kmeans' column.
    """
    with rasterio.open(in_image) as image:
        band = image.read(1)
        transform, crs, meta = image.transform, image.crs, image.meta.copy()

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(band.reshape(-1, 1))
    clusters = labels.reshape(band.shape)

    rows, cols = np.indices(band.shape)
    xs, ys = transform_xy(transform, rows.ravel(), cols.ravel())  # pixel centres
    gdf = gpd.GeoDataFrame({"cluster_kmeans": labels},
                           geometry=gpd.points_from_xy(np.asarray(xs), np.asarray(ys)), crs=crs)

    # No 'nodata' is set: cluster 0 is a valid class and must not be masked.
    meta.update({"count": 1, "dtype": "uint8", "compress": "lzw", "nodata": None})
    _ensure_parent(out_raster_lava_classification)
    with rasterio.open(out_raster_lava_classification, "w", **meta) as dst:
        dst.write(clusters.astype("uint8"), 1)
    print(f"Lava cluster GeoTIFF saved: {out_raster_lava_classification}")

    plt.figure(figsize=(12, 12))
    plt.imshow(clusters, cmap="Set1")
    plt.colorbar(label="K-Means cluster")
    _finish_plot(show)
    return gdf


def lava_extractor(lava_classified_gdf, out_shapefile_lavaflow, cluster_number=None,
                   dbscan_cluster=0, eps=100, min_samples=100, alpha=0.1, show=True):
    """
    Extract the lava flow polygon from one K-Means cluster.

    cluster_number : K-Means cluster that represents lava (asked interactively if None).
    dbscan_cluster : DBSCAN cluster to keep. 0 reproduces the original behaviour;
                     use "largest" to keep the biggest cluster instead.
    eps, min_samples : DBSCAN parameters (eps in CRS units, i.e. metres for UTM).
    alpha : alpha-shape parameter used to build the (concave) outline.
    """
    from alphashape import alphashape  # imported here: slow to import and only needed here

    if cluster_number is None:
        cluster_number = int(input("Enter the K-Means cluster number representing the lava flow: "))
    lava = lava_classified_gdf[lava_classified_gdf["cluster_kmeans"] == cluster_number].copy()
    if lava.empty:
        raise ValueError(f"K-Means cluster {cluster_number} has no pixels.")
    lava["x"], lava["y"] = lava.geometry.x, lava.geometry.y
    lava["cluster_dbscan"] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(lava[["x", "y"]])

    valid = lava[lava["cluster_dbscan"] != -1]
    if valid.empty:
        raise ValueError("DBSCAN found no cluster; try a smaller eps or min_samples.")
    keep = valid["cluster_dbscan"].value_counts().idxmax() if dbscan_cluster == "largest" else int(dbscan_cluster)
    final = lava[lava["cluster_dbscan"] == keep]
    if final.empty:
        raise ValueError(f"DBSCAN cluster {keep} does not exist; try dbscan_cluster='largest'.")

    fig, ax = plt.subplots(figsize=(8, 8))
    final.plot(ax=ax, markersize=1)
    ax.set_title("Extracted lava flow")
    _finish_plot(show)

    hull = alphashape(list(zip(final["x"], final["y"])), alpha)
    out = gpd.GeoDataFrame(geometry=[hull], crs=lava_classified_gdf.crs)
    _ensure_parent(out_shapefile_lavaflow)
    out.to_file(out_shapefile_lavaflow, driver="ESRI Shapefile")
    print(f"Lava flow saved: {out_shapefile_lavaflow}")
    if out.crs is not None and out.crs.is_projected:
        print(f"Lava flow area: {out.area.sum() / 1e6:.2f} km2")
    return out


# --------------------------------------------------------------- affected buildings / DTM
def affected_buildings(buildings_filtered_gdf, lava_polygon_gdf, out_shapefile_affected_buildings,
                       out_shapefile_affected_buildings_points):
    """
    Buildings whose footprint intersects the lava flow polygon(s).
    `lava_polygon_gdf` must contain polygons (not the per-pixel point GeoDataFrame).
    """
    if not lava_polygon_gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise TypeError("lava_polygon_gdf must contain Polygon/MultiPolygon geometries.")
    lava = lava_polygon_gdf[[lava_polygon_gdf.geometry.name]]
    if lava.crs != buildings_filtered_gdf.crs:
        lava = lava.to_crs(buildings_filtered_gdf.crs)

    hit = gpd.sjoin(buildings_filtered_gdf, lava, how="inner", predicate="intersects")
    hit = hit[~hit.index.duplicated()].drop(columns="index_right")
    print(f"Total affected buildings: {len(hit)}")

    _ensure_parent(out_shapefile_affected_buildings)
    hit["polygons"].to_file(out_shapefile_affected_buildings, driver="ESRI Shapefile")
    hit["centroids"].to_file(out_shapefile_affected_buildings_points, driver="ESRI Shapefile")
    print(f"Affected buildings saved: {out_shapefile_affected_buildings}")
    return hit


def elevation_change(in_raster_before, in_raster_after, out_raster_difference,
                     resampling=Resampling.bilinear, after_minus_before=False):
    """
    Elevation difference between two rasters on the grid of the 'before' raster.

    The 'after' raster is resampled onto the 'before' grid. By default the output is
    before - after (as in the original analysis, so lava thickness is NEGATIVE);
    set after_minus_before=True for the usual after - before convention.
    """
    with rasterio.open(in_raster_before) as src1, rasterio.open(in_raster_after) as src2:
        nodata = src1.nodata if src1.nodata is not None else -9999.0
        before = src1.read(1).astype("float32")
        after = np.full(before.shape, nodata, dtype="float32")
        rasterio.warp.reproject(
            source=rasterio.band(src2, 1), destination=after,
            src_transform=src2.transform, src_crs=src2.crs, src_nodata=src2.nodata,
            dst_transform=src1.transform, dst_crs=src1.crs, dst_nodata=nodata,
            resampling=resampling)
        valid = (before != nodata) & (after != nodata) & np.isfinite(before) & np.isfinite(after)
        diff = (after - before) if after_minus_before else (before - after)
        diff = np.where(valid, diff, nodata).astype("float32")

        profile = src1.profile
        profile.update(count=1, dtype="float32", nodata=nodata)
        _ensure_parent(out_raster_difference)
        with rasterio.open(out_raster_difference, "w", **profile) as dst:
            dst.write(diff, 1)
    print(f"Elevation change GeoTIFF saved: {out_raster_difference}")


# ------------------------------------------------------------------------------ database
def _db_settings():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "database": os.getenv("DB_NAME", "la_palma"),
        "username": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD") or getpass.getpass("Enter the database password: "),
    }


def get_db_engine(settings=None):
    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL
    s = settings or _db_settings()
    return create_engine(URL.create("postgresql+psycopg2", username=s["username"], password=s["password"],
                                    host=s["host"], port=int(s["port"]), database=s["database"]))


def import_geoDataFrame_to_database(input_gdf, table_name, if_exists="replace", engine=None):
    """
    Write a GeoDataFrame to PostGIS. Only the active geometry column is written (extra
    geometry columns such as 'centroids' are dropped). if_exists: 'replace', 'append' or 'fail'.
    """
    gdf = input_gdf.drop(columns=["index_right"], errors="ignore")
    extra = [c for c in gdf.columns if c != gdf.geometry.name and str(gdf[c].dtype) == "geometry"]
    gdf = gdf.drop(columns=extra)
    gdf.to_postgis(table_name, con=engine or get_db_engine(), if_exists=if_exists)
    print(f"{len(gdf)} rows saved to PostGIS table '{table_name}'.")


def import_raster_to_database(input_raster, table_name, srid=32628, tile_size="256x256"):
    """
    Load a raster into PostGIS with raster2pgsql | psql. Both programs must be on PATH
    (or set RASTER2PGSQL / PSQL to their full paths).
    """
    raster2pgsql = os.getenv("RASTER2PGSQL") or shutil.which("raster2pgsql")
    psql = os.getenv("PSQL") or shutil.which("psql")
    if not raster2pgsql or not psql:
        raise FileNotFoundError("raster2pgsql / psql not found. Add PostgreSQL's bin folder to PATH "
                                "or set RASTER2PGSQL and PSQL.")
    s = _db_settings()
    sql = subprocess.run([raster2pgsql, "-s", str(srid), "-I", "-C", "-M", "-t", tile_size,
                          str(input_raster), table_name], capture_output=True)
    if sql.returncode != 0:
        print("raster2pgsql failed:", sql.stderr.decode(errors="replace"))
        return
    load = subprocess.run([psql, "-h", s["host"], "-p", str(s["port"]), "-d", s["database"],
                           "-U", s["username"], "-q"],
                          input=sql.stdout, capture_output=True,
                          env={**os.environ, "PGPASSWORD": s["password"]})
    if load.returncode == 0:
        print(f"Raster {input_raster} saved to PostGIS table '{table_name}'.")
    else:
        print("psql failed:", load.stderr.decode(errors="replace"))


# ------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.getenv("LA_PALMA_DATA_DIR", "data"))
    ap.add_argument("--output-dir", default=os.getenv("LA_PALMA_OUTPUT_DIR", "output"))
    ap.add_argument("--lava-cluster", type=int, default=None,
                    help="K-Means cluster number of the lava flow (asked interactively if omitted)")
    ap.add_argument("--lava-dbscan-cluster", default="0",
                    help="DBSCAN cluster to keep: an integer (default 0) or 'largest'")
    ap.add_argument("--no-plots", action="store_true", help="do not open matplotlib windows")
    ap.add_argument("--to-db", action="store_true", help="also load results into PostGIS")
    ap.add_argument("--raster-to-db", action="store_true", help="also load the elevation-change raster")
    args = ap.parse_args(argv)

    p = build_paths(args.data_dir, args.output_dir)
    show = not args.no_plots
    dbscan_cluster = args.lava_dbscan_cluster if args.lava_dbscan_cluster == "largest" else int(args.lava_dbscan_cluster)

    lidar = import_las_to_geoDataFrame(p["aoi_las"], DESIRED_CRS)
    classified = random_forest_classifier(lidar, p["rf_model"])
    buildings = buildings_to_polygon(classified, RESOLUTION, DESIRED_CRS, p["buildings_raster"],
                                     p["buildings_polygon"], p["buildings_point"])
    lava_pixels = lava_cluster_plotter(p["image"], p["lava_raster"], show=show)
    lava_polygon = lava_extractor(lava_pixels, p["lava_flow"], cluster_number=args.lava_cluster,
                                  dbscan_cluster=dbscan_cluster, show=show)
    affected = affected_buildings(buildings, lava_polygon, p["affected_polygon"], p["affected_point"])
    elevation_change(p["dtm_before"], p["dtm_after"], p["difference"])

    if args.to_db:
        engine = get_db_engine()
        import_geoDataFrame_to_database(buildings, "buildings", engine=engine)
        import_geoDataFrame_to_database(lava_polygon, "lava_flow", engine=engine)
        import_geoDataFrame_to_database(affected, "affected_buildings", engine=engine)
    if args.raster_to_db:
        import_raster_to_database(p["difference"], "elevation_raster")


if __name__ == "__main__":
    main()
