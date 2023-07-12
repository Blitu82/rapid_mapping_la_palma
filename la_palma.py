import laspy
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.enums import Resampling
import pickle
import folium

from pathlib import Path
from shapely.geometry import Point, Polygon, MultiPolygon, shape
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from alphashape import alphashape
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sqlalchemy import create_engine
import geoalchemy2
import getpass
import subprocess

# INPUTS
IN_SAMPLE_LAS = Path(r"C:\Users\garpa\OneDrive\Documents\GitHub\final_project\La_Palma\input\las\extracted_lidar.las")
IN_AOI_LAS = Path(r"C:\Users\garpa\OneDrive\Documents\GitHub\final_project\La_Palma\input\las\lidar_AOI.las")
IN_IMAGE = Path(r"C:\Users\garpa\OneDrive\Documents\GitHub\final_project\La_Palma\Fotos aereas\sentinel_lava.tif")
IN_RASTER_BEFORE = Path(r"C:\Users\garpa\OneDrive\Documents\GitHub\final_project\La_Palma\input\DTM\before\Before_MDT02_REGCAN95_UTM28N_2m.tiff")
IN_RASTER_AFTER = Path(r"C:\Users\garpa\OneDrive\Documents\GitHub\final_project\La_Palma\input\DTM\after\Cumbre_Vieja_DSM_SfM_January_2022_20cm_REGCAN95.tif")
DESIRED_CRS = CRS.from_epsg(32628)
RESOLUTION = 1.75

# OUTPUTS
OUT_RASTER_BUILDINGS = Path(r"C:\Users\garpa\OneDrive\Documents\GitHub\final_project\La_Palma\output\output_raster\buildings.tif")
OUT_RASTER_LAVA_CLASSIFICATION = Path(r"C:\Users\garpa\OneDrive\Documents\GitHub\final_project\La_Palma\output\output_raster\classified_lava.tif")
OUT_SHAPEFILE_BUILDINGS_POLYGON = Path(r"C:\Users\garpa\OneDrive\Documents\GitHub\final_project\La_Palma\output\output_shape\buildings_rasterize\buildings_polygon.shp")
OUT_SHAPEFILE_BUILDINGS_POINT = Path(r"C:\Users\garpa\OneDrive\Documents\GitHub\final_project\La_Palma\output\output_shape\buildings_rasterize\buildings_points.shp")
OUT_SHAPEFILE_BUILDINGS_DBSCAN = Path(r"C:\Users\garpa\OneDrive\Documents\GitHub\final_project\La_Palma\output\output_shape\buildings_dbscan\buildings_dbscan.shp")
OUT_SHAPEFILE_LAVAFLOW = Path(r"C:\Users\garpa\OneDrive\Documents\GitHub\final_project\La_Palma\output\output_shape\lava_flow\lava_flow.shp")
OUT_SHAPEFILE_AFFECTED_BUILDINGS = Path(r"C:\Users\garpa\OneDrive\Documents\GitHub\final_project\La_Palma\output\output_shape\affected_buildings\affected_buildings_area.shp")
OUT_SHAPEFILE_AFFECTED_BUILDINGS_POINTS = Path(r"C:\Users\garpa\OneDrive\Documents\GitHub\final_project\La_Palma\output\output_shape\affected_buildings\affected_buildings_point.shp")
OUT_RF_MODEL = Path(r"C:\Users\garpa\OneDrive\Documents\GitHub\final_project\La_Palma\output\random_forest_classifier\RF_model.pickle")
OUT_RASTER_DIFFERENCE = Path(r"C:\Users\garpa\OneDrive\Documents\GitHub\final_project\La_Palma\output\DTM_difference\difference.tif")

# DEFINE FUNCTIONS:

def import_las_to_geoDataFrame(in_file, desired_crs):
    """
    Imports a LAS file and converts it into a Pandas GeoDataFrame.

    Parameters:
        in_file (str): The path to the LAS file to be imported.

    Returns:
        lidar_gdf (GeoDataFrame): A GeoDataFrame containing the LAS data
                                  with x, y, z, classification and geometry columns.
    """
    print("Loading LAS file...")
    las_data = laspy.read(in_file)
    x = np.array(las_data.x)
    y = np.array(las_data.y)
    z = np.array(las_data.z)
    c = np.array(las_data.classification)

    df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'classification': c})
    df['geometry'] = df.apply(lambda row: Point(row['x'], row['y']), axis=1)
    lidar_gdf = gpd.GeoDataFrame(df, crs=desired_crs, geometry=df['geometry'])
    print("LAS file loaded to a Pandas GeoDataFrame.")
    return lidar_gdf

def random_forest_classifier(lidar_gdf, model_output_path):
    """
    Trains a Random Forest Classifier using lidar data and saves the model to a pickle file.

    Parameters:
        lidar_gdf (GeoDataFrame): A GeoDataFrame containing lidar data with columns for x, y, z,
                                  classification and geometry.
        model_output_path (str): The path to save the trained Random Forest model as a pickle file.

    Returns:
        lidar_classified_gdf (GeoDataFrame): A copy of the input GeoDataFrame with an additional
                                             column ('predicted_classification') that contains the
                                             predicted classifications from the Random Forest model.
    """
    X = lidar_gdf.drop(['classification', 'geometry'], axis=1)
    y = lidar_gdf['classification']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    rf_classifier = RandomForestClassifier()
    print("Fitting Random Forest Classifier Model...")
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest accuracy: {accuracy}")
    with open(model_output_path, 'wb') as file:
        pickle.dump(rf_classifier, file)
    print(f"Random Forest Model Pickle File saved: {model_output_path}")

    lidar_classified_gdf = lidar_gdf.copy()
    lidar_classified_gdf['predicted_classification'] = rf_classifier.predict(X)

    return lidar_classified_gdf
    
def buildings_to_polygon(lidar_classified_gdf, resolution, desired_crs, output_raster, output_shape_buildings, output_shape_buildings_points):
    """
    Converts classified building points from a lidar GeoDataFrame into polygons and saves them as an ESRI Shapefile.

    Parameters:
        lidar_classified_gdf (GeoDataFrame): A GeoDataFrame containing classified lidar data with columns
                                             for x, y, z, classification, geometry and predicted_classification.
        resolution (float): The resolution (pixel size) to be used for rasterization.
        desired_crs (str): The desired CRS (Coordinate Reference System) for the output GeoDataFrames.
        output_raster (str): The path to save the rasterized building points as a GeoTIFF raster file.
        output_shape_buildings (str): The path to save the buildings polygons as an ESRI Shapefile.
        output_shape_buildings_points (str): The path to save the buildings centroids as an ESRI Shapefile.

    Returns:
        buildings_filtered_gdf (GeoDataFrame): A GeoDataFrame containing the buildings polygons
                                               with columns for polygons, centroids and area.
    """
    building_points_gdf = lidar_classified_gdf[lidar_classified_gdf['predicted_classification'] == 6]
    bounds = building_points_gdf.total_bounds
    rows = int((bounds[3] - bounds[1]) / resolution)
    cols = int((bounds[2] - bounds[0]) / resolution)

    output_profile = {
        'driver': 'GTiff',
        'dtype': np.uint8,
        'count': 1,
        'width': cols,
        'height': rows,
        'transform': from_bounds(*bounds, cols, rows),
        'crs': DESIRED_CRS,
    }

    raster = np.zeros((output_profile['height'], output_profile['width']), dtype=np.uint8)

    rasterized = rasterize(
        shapes=building_points_gdf.geometry,
        out=raster,
        transform=output_profile['transform'],
        fill=0,
        default_value=255,
        dtype=np.uint8,
    )

    with rasterio.open(output_raster, 'w', **output_profile) as dst:
        dst.write(rasterized, 1)

    with rasterio.open(output_raster) as src:
        raster = src.read(1)
        transform = src.transform
    print(f"Building Raster saved: {output_raster}")

    polygons = []
    for polygon, value in shapes(raster, mask=raster == 255, transform=transform):
        if value == 255:
            polygons.append(shape(polygon))

    buildings_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=desired_crs)
    buildings_gdf = buildings_gdf.rename_geometry('polygons')
    buildings_gdf['centroids'] = buildings_gdf.centroid
    buildings_gdf['area'] = buildings_gdf.area

    buildings_filtered_gdf = buildings_gdf[buildings_gdf.area > 10]
    buildings_filtered_gdf['polygons'].to_file(output_shape_buildings, driver='ESRI Shapefile')
    buildings_filtered_gdf['centroids'].to_file(output_shape_buildings_points, driver='ESRI Shapefile')
    print(f"Building Shapefiles saved: {output_shape_buildings}, {output_shape_buildings_points}")
    return buildings_filtered_gdf

def buildings_to_polygon_dbscan(lidar_classified_gdf, desired_crs, out_shapefile_buildings_dbscan):
    """
    Converts classified building points from a lidar GeoDataFrame into polygons using DBSCAN algorithm and saves them as an ESRI Shapefile.

    Parameters:
        lidar_classified_gdf (GeoDataFrame): A GeoDataFrame containing classified lidar data with columns
                                             for x, y, z, classification, geometry and predicted_classification.
        desired_crs (str): The desired CRS (Coordinate Reference System) for the output GeoDataFrame.
        out_shapefile_buildings_dbscan (str): The path to save the buildings polygons generated by DBSCAN as an ESRI Shapefile.

    Returns:
        buildings_polygon_gdf (GeoDataFrame): A GeoDataFrame containing the polygons of the buildings generated by DBSCAN
                                              with the cluster labels.
    """
    gdf_dbscan = lidar_classified_gdf[lidar_classified_gdf['predicted_classification'] == 6]
    gdf_dbscan = gdf_dbscan[['x', 'y', 'geometry']]
    coords = gdf_dbscan[['x', 'y']].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    eps = 0.1
    min_samples = 5
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(coords_scaled)

    gdf_dbscan['cluster'] = clusters

    polygons = []

    gdf_clusters = gdf_dbscan[gdf_dbscan['cluster'] != -1]

    for cluster_label in gdf_clusters['cluster'].unique():
        cluster_coords = gdf_clusters.loc[gdf_clusters['cluster'] == cluster_label, ['x', 'y']].values
        outline_polygon = Polygon(cluster_coords).convex_hull
        polygons.append(outline_polygon)

    buildings_polygon_gdf = gpd.GeoDataFrame(geometry=polygons, index=gdf_clusters['cluster'].unique(), crs=desired_crs)

    buildings_polygon_gdf.to_file(out_shapefile_buildings_dbscan, driver='ESRI Shapefile')
    print(f"Building Shapefiles saved: {out_shapefile_buildings_dbscan}")
    return buildings_polygon_gdf

def lava_cluster_plotter(in_image, out_raster_lava_classification):
    """
    Perform K-means clustering on a satellite image and save the cluster image as a GeoTIFF file.

    Parameters:
        in_image (str): Path to the input raster image.
        out_raster_lava_classification (str): Path to save the output cluster image as a GeoTIFF file.

    Returns:
        lava_classified_gdf (GeoDataFrame): GeoDataFrame containing the K-Means cluster assignments.
    """
    
    image = rasterio.open(in_image)
    band_index = 1  
    band = image.read(band_index)

    X = band.reshape((-1,1))

    k_means = KMeans(n_clusters=10)
    k_means.fit(X)

    clusters = k_means.labels_
    clusters = clusters.reshape(band.shape)
    
    coords = []
    for i in range(image.height):
        for j in range(image.width):
            lon, lat = image.xy(i, j)
            coords.append(Point(lon, lat))
        
    lava_classified_df = pd.DataFrame({'geometry': coords, 'cluster_kmeans': clusters.flatten()})

    lava_classified_gdf = gpd.GeoDataFrame(lava_classified_df, crs=image.crs)

    metadata = image.meta
    metadata.update({
        'count': 1,  
        'dtype': rasterio.uint8, 
        'compress': 'lzw',  
         'nodata': 0,  
    })

    with rasterio.open(out_raster_lava_classification, 'w', **metadata) as dst:
        dst.write(clusters.astype(rasterio.uint8), 1)
    print(f'Lava Clusters GeoTIFF file saved: {out_raster_lava_classification}')

    plt.figure(figsize=(20,20))
    plt.imshow(clusters, cmap="Set1")
    plt.colorbar(label='Classification')
    plt.show()
    
    return lava_classified_gdf
        
def lava_extractor(lava_classified_gdf, out_shapefile_lavaflow):
    """
    Extract a lava flow polygon from a cluster in a GeoDataFrame using DBSCAN clustering.

    Parameters:
        lava_classified_gdf (GeoDataFrame): GeoDataFrame containing cluster assignments and coordinates.
        out_shapefile_lavaflow (str): Path to save the output shapefile containing the lava flow polygon.

    Returns:
        lava_polygon_gdf (GeoDataFrame): GeoDataFrame containing the extracted lava flow polygon.
    """
    cluster_number = int(input('Enter the K-Means cluster number representing the lava flow:'))
    lava_gdf = lava_classified_gdf[lava_classified_gdf['cluster_kmeans'] == cluster_number].copy()
    lava_gdf['x'] = lava_gdf['geometry'].x
    lava_gdf['y'] = lava_gdf['geometry'].y
    
    dbscan = DBSCAN(eps=100, min_samples=100)
    labels = dbscan.fit_predict(lava_gdf[['x', 'y']])
    lava_gdf['cluster_dbscan'] = labels
    
    fig, ax = plt.subplots(figsize=(10, 10))
    lava_gdf.plot(column='cluster_dbscan', cmap='Set1', ax=ax, legend=True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('DBSCAN Clustering')
    plt.show()
    
    final_lava_gdf = lava_gdf[lava_gdf['cluster_dbscan'] == 0]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    final_lava_gdf.plot(column='cluster_dbscan', cmap='Set1', ax=ax, legend=True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Extracted Lava Flow')
    plt.show()
    
    points = final_lava_gdf.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()
    
    alpha = 0.1  
    concave_hull = alphashape(points, alpha)
    lava_polygon_gdf = gpd.GeoDataFrame(geometry=[concave_hull], crs=lava_classified_gdf.crs)
    lava_polygon_gdf.to_file(out_shapefile_lavaflow, driver='ESRI Shapefile')
    print(f'Lava Flow Shapefile saved: {out_shapefile_lavaflow}')
    
    return lava_polygon_gdf

def affected_buildings(buildings_filtered_gdf, lava_polygon_gdf, out_shapefile_affected_buildings, out_shapefile_affected_buildings_points):
    """
    Identify the buildings affected by a lava flow and save them as an ESRI Shapefile.

    Parameters:
        buildings_filtered_gdf (GeoDataFrame): GeoDataFrame containing the buildings to be analyzed.
        lava_polygon_gdf (GeoDataFrame): GeoDataFrame representing the lava flow polygon.
        out_shapefile_affected_buildings (str): Path to save the output ESRI Shapefile containing the affected building polygons.
        out_shapefile_affected_buildings_points (str): Path to save the output ESRI Shapefile containing the centroids of the affected buildings.

    Returns:
        buildings_w_lava_gdf (GeoDataFrame): GeoDataFrame containing the affected buildings.
    """

    buildings_w_lava_gdf = gpd.sjoin(buildings_filtered_gdf, lava_polygon_gdf, predicate="intersects")
    print(f"Total affected buildings: {len(buildings_w_lava_gdf)}")
    buildings_w_lava_gdf['polygons'].to_file(out_shapefile_affected_buildings, driver='ESRI Shapefile')
    buildings_w_lava_gdf['centroids'].to_file(out_shapefile_affected_buildings_points, driver='ESRI Shapefile')
    print(f'Affected Buildings Shapefiles saved: {out_shapefile_affected_buildings} , {out_shapefile_affected_buildings_points}')
    return buildings_w_lava_gdf 

def elevation_change(in_raster_before, in_raster_after, out_raster_difference):
    """
    Calculate the elevation change between two input raster files and save the difference as a GeoTIFF.

    Parameters:
        in_raster_before (Path): Path to the input raster file representing elevation data before the event.
        in_raster_after (Path): Path to the input raster file representing elevation data after the event.
        out_raster_difference (Path): Path to save the output GeoTIFF file containing the elevation change.

    """
    src1 = rasterio.open(in_raster_before)
    src2 = rasterio.open(in_raster_after)

    # Reproject the second raster to match the resolution and extent of the first raster
    data2, _ = rasterio.warp.reproject(
        source=rasterio.band(src2, 1),
        destination=np.empty_like(src1.read(1)),
        src_transform=src2.transform,
        src_crs=src2.crs,
        dst_transform=src1.transform,
        dst_crs=src1.crs,
        resampling=Resampling.bilinear)

    # Read the input raster data
    data1 = src1.read(1)

    # Compute the difference between the rasters
    diff = data1 - data2

    # Replace inf and NaN values with a valid nodata value
    diff[np.isinf(diff) | np.isnan(diff)] = src1.nodata

    # Prepare the output GeoTIFF file
    output_profile = src1.profile
    output_profile.update(count=1)  # Update band count

    # Create the output file
    with rasterio.open(out_raster_difference, 'w', **output_profile) as dst:
        # Write the difference data to the output file
        dst.write(diff, 1)
    print(f'Elevation Change GeoTIFF saved: {out_raster_difference}')

    # Close the input files
    src1.close()
    src2.close()

def import_geoDataFrame_to_database(input_gdf, table_name):
    """
    Imports a GeoDataFrame into a PostGIS database table.

    Parameters:
        input_gdf (GeoDataFrame): The GeoDataFrame containing the spatial data to be imported.
        table_name (str): The name of the PostGIS database table to import the data into.

    Returns:
        None
    """
    password = getpass.getpass('Enter the database password:')
    engine = create_engine('postgresql://postgres:' + password + '@localhost:5432/la_palma')
    input_gdf.to_postgis(table_name, con=engine)
    print(f'GeoDataFrame {input_gdf} saved to the PostGIS database table {table_name}.')

def import_raster_to_database(input_raster, table_name):
    """
    Imports a raster dataset into a PostGIS database table.

    Parameters:
        input_raster (str): The file path to the raster dataset to be imported.
        table_name (str): The name of the PostGIS database table to import the raster into.

    Returns:
        None

    """
    host = 'localhost'
    port = '5432'
    database = 'la_palma'
    username = 'postgres'
    password = getpass.getpass('Enter the database password:')

    # Define the command to run, including any necessary arguments
    command_raster2pgsql = [
        r"C:\Program Files\PostgreSQL\15\bin\raster2pgsql.exe", '-I', '-C', '-M', input_raster
    ]
    command_psql = [
        r"C:\Program Files\PostgreSQL\15\bin\psql.exe", '-h', host, '-p', port, '-d', database, '-U', username,
        '--password', password
    ]

    # Run the raster2pgsql command and capture the output
    process_raster2pgsql = subprocess.Popen(command_raster2pgsql, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output_raster2pgsql, error_raster2pgsql = process_raster2pgsql.communicate()

    # Check the return code of raster2pgsql command
    if process_raster2pgsql.returncode == 0:
        # Run the psql command and capture the output
        process_psql = subprocess.Popen(command_psql, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output_psql, error_psql = process_psql.communicate(input=output_raster2pgsql)
        
        # Check the return code of psql command
        if process_psql.returncode == 0:
            print(f'Raster file {input_raster} saved to the PostGIS database table {table_name}')
        else:
            print('An error occurred in psql command:', error_psql.decode())
    else:
        print('An error occurred in raster2pgsql command:', error_raster2pgsql.decode())  

def main():
    """
    Executes the main workflow for processing lidar and image data.

    Parameters:
        None

    Returns:
        None
    """
    lidar_gdf = import_las_to_geoDataFrame(IN_AOI_LAS, DESIRED_CRS)
    lidar_classified_gdf = random_forest_classifier(lidar_gdf, OUT_RF_MODEL)
    buildings_filtered_gdf = buildings_to_polygon(lidar_classified_gdf, RESOLUTION, DESIRED_CRS, OUT_RASTER_BUILDINGS, OUT_SHAPEFILE_BUILDINGS_POLYGON, OUT_SHAPEFILE_BUILDINGS_POINT)
    lava_classified_gdf = lava_cluster_plotter(IN_IMAGE, OUT_RASTER_LAVA_CLASSIFICATION) 
    lava_polygon_gdf = lava_extractor(lava_classified_gdf, OUT_SHAPEFILE_LAVAFLOW)
    buildings_w_lava_gdf = affected_buildings(buildings_filtered_gdf, lava_classified_gdf, OUT_SHAPEFILE_AFFECTED_BUILDINGS, OUT_SHAPEFILE_AFFECTED_BUILDINGS_POINTS)
    elevation_change(IN_RASTER_BEFORE, IN_RASTER_AFTER, OUT_RASTER_DIFFERENCE)
    import_geoDataFrame_to_database(buildings_filtered_gdf, 'buildings')
    import_geoDataFrame_to_database(lava_polygon_gdf, 'lava_flow')
    import_geoDataFrame_to_database(buildings_w_lava_gdf, 'affected_buildings')
    # import_raster_to_database(OUT_RASTER_DIFFERENCE, 'elevation_raster')

if __name__ == '__main__':
    main()