# Rapid Mapping: La Palma volcanic eruption

Python pipeline that estimates the **extent of the 2021 Cumbre Vieja lava flow**, the **buildings affected** and the **terrain change**, using airborne lidar, a Sentinel satellite image and pre/post-eruption elevation models. Results are stored as shapefiles/GeoTIFFs and, optionally, in a **PostGIS** database.

## Key results

| Output | Value |
|---|---|
| Lava flow extent | ~12.4 km² |
| Affected buildings | 3,126 |
| Building classification (Random Forest) | ~86 % accuracy* |

\*Measured on a random hold-out of the same lidar point cloud, see [Limitations](#limitations-and-next-steps).

<!--
  TO DO: paste here the image lines that are in your current README
  (lava flow map, affected buildings, elevation change), e.g.
  ![Lava flow extent](images/your_image.png)
-->

## Method

1. **Buildings from lidar.** A Random Forest is trained on the lidar point cloud (x, y, z) to predict the point classification. Points predicted as *building* (class 6) are rasterised (1.75 m cells), vectorised into polygons, and polygons under 10 m² are discarded. An alternative DBSCAN + convex-hull extraction is also provided.
2. **Lava flow from Sentinel.** K-Means (10 clusters) is applied to a Sentinel band. The cluster that represents lava is selected, DBSCAN removes isolated pixels, and an alpha shape gives the flow outline.
3. **Affected buildings.** Spatial join of building footprints with the lava polygon.
4. **Elevation change.** The post-eruption surface model (Jan 2022, 20 cm) is resampled onto the pre-eruption DTM grid (2 m) and the two are differenced.
5. **Storage.** Buildings, lava flow and affected buildings are written to PostGIS tables.

## Repository structure

```
.
├── la_palma.py              # the pipeline (command line)
├── tests/test_la_palma.py   # tests on synthetic data
├── requirements.txt         # pip dependencies
├── environment.yml          # conda dependencies (recommended on Windows)
├── docker-compose.yml       # local PostGIS database
├── .env.example             # database settings template
└── ...                      # notebook, presentation and images
```

## Getting started

### 1. Install

```bash
conda env create -f environment.yml
conda activate la-palma
# or: python -m venv .venv && pip install -r requirements.txt
```

### 2. Get the data

Create a `data/` folder with this layout (files are not stored in the repository):

```
data/
├── las/lidar_AOI.las
├── sentinel/sentinel_lava.tif
└── dtm/
    ├── before/Before_MDT02_REGCAN95_UTM28N_2m.tiff
    └── after/Cumbre_Vieja_DSM_SfM_January_2022_20cm_REGCAN95.tif
```

Sources: Spanish national lidar (PNOA-LiDAR, CNIG), Copernicus Sentinel imagery, and the pre/post-eruption elevation models as described in the original project. All data use ETRS89/REGCAN95 and are processed in WGS 84 / UTM 28N (EPSG:32628).

### 3. Run

```bash
python la_palma.py --data-dir data --output-dir output
```

The script shows the K-Means clusters and asks which cluster is the lava flow. To skip the prompt and plots:

```bash
python la_palma.py --lava-cluster 3 --no-plots
```

Outputs go to `output/` (shapefiles, GeoTIFFs, trained model).

### 4. (Optional) PostGIS

```bash
cp .env.example .env          # then edit DB_PASSWORD
docker compose up -d          # starts PostGIS on localhost:5432
python la_palma.py --to-db    # writes buildings, lava_flow, affected_buildings
```

Tables are replaced on each run. The password is read from `.env` / environment, or asked interactively. Never commit `.env`.

### Tests

```bash
pytest
```

## Limitations and next steps

- **Random Forest evaluation.** The model uses only x, y, z and is scored on a random 20 % split of the same point cloud, so the ~86 % accuracy is likely optimistic for new areas. A spatial hold-out, extra features (return number, intensity, local height) and per-class precision/recall would be a fairer test.
- **Lava extraction.** K-Means on a single band is sensitive to clouds, ash and shadows, and the lava cluster is chosen by eye. Comparing the 12.4 km² with the Copernicus EMS delineation would validate it.
- **Elevation change.** The default sign is *before − after*, so new lava appears negative. Use `elevation_change(..., after_minus_before=True)` for the usual convention.
- **Coordinates and DBSCAN.** The DBSCAN building extraction uses standardised coordinates, so `eps` depends on the study area extent.

## Author

Blitu82 · [GitHub profile](https://github.com/Blitu82)

## License

MIT, see [LICENSE](LICENSE).
