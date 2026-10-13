# Rapid Mapping of the 2021 La Palma Volcanic Eruption

Automated geospatial pipeline in Python that estimates the **extent of the lava flow**, the **buildings affected** and the **change in the island's morphology** after the 2021 Cumbre Vieja eruption (La Palma, Canary Islands), and stores the results in a **PostgreSQL / PostGIS** database.

Final project of the Ironhack Data Analytics bootcamp.

## Background

The Cumbre Vieja volcano (La Palma, Canary Islands, Spain) began erupting on 19 September 2021. When the activity ended on 13 December 2021, the island's morphology had changed and thousands of buildings had been buried by lava. During the eruption, the European Union's [Copernicus Emergency Management Service](https://emergency.copernicus.eu/) provided rapid mapping of the lava flow and the damage to buildings and infrastructure.

This project is inspired by that work and aims to automate the production of similar products from open lidar and satellite data.

## Objectives

1. Extract the **extent of the lava flow** from satellite imagery.
2. Detect buildings from lidar and count those **affected by the lava flow**.
3. Measure the **change in the island's morphology** caused by the lava.
4. Store the results in a **PostgreSQL / PostGIS** database.

## Method

1. **Lidar import:** LAS point clouds are read with `laspy` and converted to a GeoDataFrame (x, y, z, classification).
2. **Building detection:** a Random Forest classifier (scikit-learn) is trained on the lidar points. Points classified as buildings (class 6) are rasterised (1.75 m resolution), vectorised into polygons and filtered by area. An alternative DBSCAN-based approach is included in the notebook.
3. **Lava extent:** a K-Means classification (10 clusters) is applied to a Sentinel satellite image. The cluster representing lava is selected, noise is removed with DBSCAN, and a concave hull (alpha shape) gives the lava flow polygon.
4. **Affected buildings:** a spatial join (intersects) between the building polygons and the lava flow polygon.
5. **Morphology change:** the post-eruption surface model (January 2022, 20 cm) is resampled to the pre-eruption terrain model grid and subtracted to produce an elevation difference raster.
6. **Database:** buildings, lava flow and affected buildings are loaded into PostGIS (tables `buildings`, `lava_flow`, `affected_buildings`). A helper to load the elevation raster with `raster2pgsql` is also included.

Coordinate reference system: ETRS89 / UTM zone 28N (EPSG:32628).

## Data

The data are not included in this repository because of their size. They can be downloaded from:

- **CNIG / IGN (Spain):** lidar point clouds and digital terrain models. <https://centrodedescargas.cnig.es/CentroDescargas/index.jsp>
- **Copernicus Emergency Management Service, EMSN112:** ground deformation in La Palma. <https://emergency.copernicus.eu/mapping/list-of-components/EMSN112>
- **Sentinel imagery** of the lava flow.

## Tools

Python, Jupyter Notebook, laspy, NumPy, pandas, GeoPandas, Shapely, rasterio, scikit-learn (Random Forest, K-Means, DBSCAN), alphashape, Matplotlib, folium, SQLAlchemy, GeoAlchemy2, PostgreSQL / PostGIS.

## Author

Pablo García García
