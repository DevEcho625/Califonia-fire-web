# wildfire_risk_notebook.py
# Notebook-style script: paste into a Jupyter cell or run as a .py (line-by-line)

# --- 0. Install / import (uncomment if using in a clean env) ---
# !pip install geopandas shapely rtree fiona pyproj scikit-learn matplotlib tqdm requests

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm

# --- 1. Paths & data sources ---
WFIGS_CSV = r"C:\Users\anujx\Downloads\WFIGS_Incident_Locations_-835549102613066266 (2).csv"

CALFIRE_GEOJSON_URLS = [
    "https://data.ca.gov/dataset/california-fire-perimeters-all/resource/dd5e4337-8679-4d64-bd90-b9df85ee6b58/download", 
    "https://services.arcgis.com/jIL9msH9OI208GCb/ArcGIS/rest/services/California_Fire_Perimeters_1878_2019/FeatureServer/1/query?where=1%3D1&outFields=*&f=geojson",
    "https://gis.data.cnra.ca.gov/arcgis/rest/services/CALFIRE-Forestry/california-fire-perimeters-all/FeatureServer/0/query?where=1%3D1&outFields=*&f=geojson"
]

def load_calfire_geojson_try(urls=CALFIRE_GEOJSON_URLS):
    """Try multiple sources until CAL FIRE GeoJSON loads."""
    for url in urls:
        try:
            print("Trying:", url)
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            ct = r.headers.get("Content-Type", "")
            if "json" in ct or "geojson" in ct or r.text.strip().startswith("{"):
                data = r.json()
                gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
                print("Loaded GeoJSON from:", url)
                return gdf
            else:
                print("Response not GeoJSON ({}): trying geopandas read_file fallback".format(ct))
                try:
                    gdf = gpd.read_file(url)
                    return gdf
                except Exception as e:
                    print("geopandas read_file failed for", url, "error:", e)
        except Exception as e:
            print("Failed to load from", url, ":", e)
    raise RuntimeError("Could not automatically download CAL FIRE perimeters.")

# --- 2. Load WFIGS incidents CSV ---
print("Loading WFIGS incidents CSV...")
wf_df = pd.read_csv(WFIGS_CSV, low_memory=False)
print("Columns in WFIGS CSV:", wf_df.columns.tolist())

# Try to find lat/lon columns
lat_col = None
lon_col = None
for c in ["Latitude", "LAT", "lat", "Y", "POINT_Y", "InitialLatitude"]:
    if c in wf_df.columns:
        lat_col = c
        break
for c in ["Longitude", "LON", "lon", "X", "POINT_X", "InitialLongitude"]:
    if c in wf_df.columns:
        lon_col = c
        break

# Fallback explicit assignment
if lat_col is None or lon_col is None:
    print("Could not auto-detect lat/lon columns. Available columns:", wf_df.columns)
    lat_col = "InitialLatitude" if "InitialLatitude" in wf_df.columns else None
    lon_col = "InitialLongitude" if "InitialLongitude" in wf_df.columns else None

if lat_col is None or lon_col is None:
    raise RuntimeError("No latitude/longitude columns found. Please verify CSV headers.")

print(f"Using latitude column: {lat_col}, longitude column: {lon_col}")

# Drop missing coords and convert to GeoDataFrame
wf_df = wf_df.dropna(subset=[lat_col, lon_col]).copy()
wf_gdf = gpd.GeoDataFrame(
    wf_df,
    geometry=[Point(xy) for xy in zip(wf_df[lon_col], wf_df[lat_col])],
    crs="EPSG:4326"
)
print("Loaded WFIGS points:", len(wf_gdf))

# --- 3. Load CAL FIRE perimeters ---


print("Downloading/reading CAL FIRE perimeters (FRAP)...")
calfire_url = "https://services.arcgis.com/jIL9msH9OI208GCb/ArcGIS/rest/services/California_Fire_Perimeters_1878_2019/FeatureServer/1/query?where=1%3D1&outFields=*&f=geojson"
calfire_gdf = gpd.read_file(calfire_url)

# Clean invalid geometries
calfire_gdf = calfire_gdf[~calfire_gdf.geometry.is_empty]
calfire_gdf = calfire_gdf[calfire_gdf.geometry.notnull()]
calfire_gdf["geometry"] = calfire_gdf["geometry"].buffer(0)
calfire_gdf = calfire_gdf[calfire_gdf.is_valid]

# Reproject
target_crs = "EPSG:3310"
calfire_gdf = calfire_gdf.to_crs(target_crs)

# Verify bounds
print("CAL FIRE perimeters loaded. Features:", len(calfire_gdf))
print("Bounding box:", calfire_gdf.total_bounds)


# --- 4. Generate grid for modeling ---
minx, miny, maxx, maxy = wf_gdf.total_bounds
cell_size = 1000  # meters (1 km)

minx = -90



print(maxx, minx, maxy, miny)
nx = int((maxx - minx) / cell_size)
ny = int((maxy - miny) / cell_size)
print(f"Grid size: {nx} × {ny}")

grid_polys, grid_centroids = [], []
for i in range(nx):
    for j in range(ny):
        x0, y0 = minx + i*cell_size, miny + j*cell_size
        x1, y1 = x0 + cell_size, y0 + cell_size
        grid_polys.append(box(x0, y0, x1, y1))
        grid_centroids.append(((x0+x1)/2, (y0+y1)/2))

grid_gdf = gpd.GeoDataFrame(geometry=grid_polys, crs=target_crs)
grid_gdf["centroid"] = [Point(c) for c in grid_centroids]

# Clip grid to area around fires
study_buffer = 5000
wf_bounds = wf_gdf.total_bounds
study_box = box(wf_bounds[0]-study_buffer, wf_bounds[1]-study_buffer,
                wf_bounds[2]+study_buffer, wf_bounds[3]+study_buffer)
grid_gdf = grid_gdf[grid_gdf.intersects(study_box)].reset_index(drop=True)
print("Grid cells after clipping:", len(grid_gdf))

# --- 5. Compute features ---
calfire_gdf["perim_area"] = calfire_gdf.geometry.area
calfire_sindex = calfire_gdf.sindex

def compute_burned_area_for_cell(cell_geom):
    possible_idx = list(calfire_sindex.intersection(cell_geom.bounds))
    return sum(
        calfire_gdf.geometry.iloc[i].intersection(cell_geom).area
        for i in possible_idx
        if calfire_gdf.geometry.iloc[i].intersects(cell_geom)
    )

grid_gdf["burned_area_m2"] = [
    compute_burned_area_for_cell(g) for g in tqdm(grid_gdf.geometry, desc="computing burned area")
]
grid_gdf["burned_area_km2"] = grid_gdf["burned_area_m2"] / 1e6

def count_perimeters_in_cell(cell_geom):
    possible_idx = list(calfire_sindex.intersection(cell_geom.bounds))
    return sum(
        calfire_gdf.geometry.iloc[i].intersects(cell_geom)
        for i in possible_idx
    )

grid_gdf["burn_count"] = [
    count_perimeters_in_cell(g) for g in tqdm(grid_gdf.geometry, desc="counting perimeters")
]

# Distance to nearest perimeter
centroids = gpd.GeoSeries([p for p in grid_gdf.centroid], crs=target_crs)
nearest_distances = []
for c in tqdm(centroids, desc="computing distance"):
    possible_idx = list(calfire_sindex.nearest(c.bounds, num_results=5))
    dmin = np.min([c.distance(calfire_gdf.geometry.iloc[i]) for i in possible_idx]) if possible_idx else np.inf
    nearest_distances.append(dmin if np.isfinite(dmin) else 0)
grid_gdf["dist_to_perimeter_m"] = nearest_distances

# Label fire occurrence
wf_sindex = wf_gdf.sindex
labels = []
for geom in tqdm(grid_gdf.geometry, desc="labeling cells"):
    possible = list(wf_sindex.intersection(geom.bounds))
    labels.append(
        any(wf_gdf.geometry.iloc[i].within(geom) for i in possible)
    )
grid_gdf["label_fire"] = np.array(labels, dtype=int)

# --- 6. Model or heuristic scoring ---
feature_cols = ["burned_area_km2", "burn_count", "dist_to_perimeter_m"]
X = grid_gdf[feature_cols].fillna(0).values
print(X)
y = grid_gdf["label_fire"].values

# Invert distance to represent proximity
X[:, 2] = 1.0 / (1.0 + (X[:, 2] / 1000.0))

scaler = StandardScaler()
print(scaler)
X_scaled = scaler.fit_transform(X)

if y.sum() == 0:
    print("No labeled fires found — using heuristic risk score.")
    wa, wb, wc = 0.5, 0.3, 0.2
    raw = wa*X[:,0]/(X[:,0].max()+1e-9) + wb*X[:,1]/(X[:,1].max()+1e-9) + wc*X[:,2]/(X[:,2].max()+1e-9)
    risk = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    grid_gdf["fire_likelihood"] = risk
else:
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, stratify=y, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_scaled)[:,1]
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    print("Model ROC AUC:", auc)
    grid_gdf["fire_likelihood"] = proba

# --- 7. Output and visualize ---
out_geojson = "grid_fire_likelihood.geojson"
out_csv = "grid_fire_likelihood.csv"
grid_gdf.to_file(out_geojson, driver="GeoJSON")

grid_out = grid_gdf.copy()
grid_out["centroid_x"] = [p.x for p in grid_out.centroid]
grid_out["centroid_y"] = [p.y for p in grid_out.centroid]
grid_out[["centroid_x", "centroid_y", "burned_area_km2", "burn_count", "dist_to_perimeter_m", "fire_likelihood"]].to_csv(out_csv, index=False)
print("Saved:", out_csv, out_geojson)

fig, ax = plt.subplots(figsize=(10,10))
calfire_gdf.plot(ax=ax, linewidth=0.2, edgecolor="gray")
grid_gdf.plot(column="fire_likelihood", ax=ax, alpha=0.8, legend=True)
wf_gdf.plot(ax=ax, markersize=5, color="black")
plt.title("Predicted Fire Likelihood (0–1)")
plt.axis("off")
plt.show()

print("Risk stats: min {:.3f}, mean {:.3f}, max {:.3f}".format(
    grid_gdf["fire_likelihood"].min(),
    grid_gdf["fire_likelihood"].mean(),
    grid_gdf["fire_likelihood"].max()
))
