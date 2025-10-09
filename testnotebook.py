# wildfire_risk_notebook.py
# Notebook-style script: paste into a Jupyter cell or run as a .py (line-by-line)

# --- 0. Install / import (uncomment if using in a clean env) ---
# !pip install geopandas shapely rtree fiona pyproj scikit-learn matplotlib

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import requests
import json
from tqdm import tqdm

# --- 1. Paths & data sources ---
# Your uploaded WFIGS incidents CSV (already in this environment)
WFIGS_CSV = "/mnt/data/WFIGS_Incident_Locations_-835549102613066266.csv"

# Preferred: CAL FIRE GeoJSON from California open-data (FRAP). We try a few endpoints.
# (1) data.ca.gov resource (GeoJSON) — public FRAP dataset (fallback).
CALFIRE_GEOJSON_URLS = [
    # Resource from data.ca.gov (example resource id) - may be available as direct download
    "https://data.ca.gov/dataset/california-fire-perimeters-all/resource/dd5e4337-8679-4d64-bd90-b9df85ee6b58/download", 
    # ArcGIS FeatureServer -> query GeoJSON (one of the public endpoints discovered)
    "https://services.arcgis.com/jIL9msH9OI208GCb/ArcGIS/rest/services/California_Fire_Perimeters_1878_2019/FeatureServer/1/query?where=1%3D1&outFields=*&f=geojson",
    # Another ArcGIS REST candidate (other public services exist; this is a robust fallback)
    "https://gis.data.cnra.ca.gov/arcgis/rest/services/CALFIRE-Forestry/california-fire-perimeters-all/FeatureServer/0/query?where=1%3D1&outFields=*&f=geojson"
]

def load_calfire_geojson_try(urls=CALFIRE_GEOJSON_URLS):
    for url in urls:
        try:
            print("Trying:", url)
            # Some data.ca.gov endpoints redirect; requests can follow redirects.
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            # If the response looks like HTML (portal page), skip
            ct = r.headers.get("Content-Type", "")
            if "json" in ct or "geojson" in ct or r.text.strip().startswith("{"):
                # attempt to load as geojson
                data = r.json()
                gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
                print("Loaded GeoJSON from:", url)
                return gdf
            else:
                # Sometimes data.ca.gov returns a zipped shapefile — try to save and read if so.
                # Save to tempfile and try geopandas to read; skip complex handling for now.
                print("Response not GeoJSON (Content-Type={}): trying geopandas read_file as fallback".format(ct))
                try:
                    gdf = gpd.read_file(url)
                    return gdf
                except Exception as e:
                    print("geopandas read_file failed for url:", url, "error:", e)
                    continue
        except Exception as e:
            print("Failed to load from", url, ":", e)
            continue
    raise RuntimeError("Could not automatically download CAL FIRE perimeters. If this fails, manually download the FRAP perimeter GeoJSON/shapefile from CAL FIRE / data.ca.gov and set CALFIRE_GEOJSON_PATH to a local file.")

# --- 2. Load datasets ---
print("Loading WFIGS incidents CSV...")
wf_df = pd.read_csv(WFIGS_CSV)
# Inspect common column names to find lat/lon
print("Columns in WFIGS CSV:", list(wf_df.columns)[:30])

# Try common column names
lat_col = None
lon_col = None
for c in ["Latitude", "LAT", "lat", "Y", "POINT_Y"]:
    if c in wf_df.columns:
        lat_col = c
        break
for c in ["Longitude", "LON", "lon", "X", "POINT_X"]:
    if c in wf_df.columns:
        lon_col = c
        break

if lat_col is None or lon_col is None:
    # try parsing a single geometry column if present
    print("Could not auto-detect lat/lon columns. Available columns:", wf_df.columns)
    raise RuntimeError("Please ensure the WFIGS CSV has latitude/longitude columns or supply them.")

# Drop missing coords & convert to GeoDataFrame
wf_df = wf_df.dropna(subset=[lat_col, lon_col]).copy()
wf_gdf = gpd.GeoDataFrame(wf_df, geometry=[Point(xy) for xy in zip(wf_df[lon_col], wf_df[lat_col])], crs="EPSG:4326")
print("Loaded WFIGS points:", len(wf_gdf))

# Load CAL FIRE perimeters (attempt download)
print("Downloading/reading CAL FIRE perimeters (FRAP)... This may take 10-60s)")
calfire_gdf = load_calfire_geojson_try()
print("CAL FIRE perimeters loaded. Features:", len(calfire_gdf))

# Ensure both in same CRS (project to EPSG:3310 or keep EPSG:4326)
target_crs = "EPSG:3310"  # California Albers Equal Area (good for area calculations) — optional
wf_gdf = wf_gdf.to_crs(target_crs)
calfire_gdf = calfire_gdf.to_crs(target_crs)

# --- 3. Build features for modeling ---
# We'll create a prediction grid across the CAL FIRE perimeter bounds (or limit to WFIGS bounds)
combined_bounds = wf_gdf.total_bounds if len(wf_gdf)>0 else calfire_gdf.total_bounds
minx, miny, maxx, maxy = combined_bounds
print("Study bounds in CRS {}:".format(target_crs), combined_bounds)

# Build regular grid (grid cell = 1km x 1km by default)
cell_size = 1000  # meters
nx = int(np.ceil((maxx - minx) / cell_size))
ny = int(np.ceil((maxy - miny) / cell_size))
print(f"Grid size: {nx} x {ny} cells (approximately)")

grid_polys = []
grid_centroids = []
for i in range(nx):
    for j in range(ny):
        x0 = minx + i*cell_size
        y0 = miny + j*cell_size
        x1 = x0 + cell_size
        y1 = y0 + cell_size
        grid_polys.append(box(x0, y0, x1, y1))
        grid_centroids.append(((x0+x1)/2, (y0+y1)/2))

grid_gdf = gpd.GeoDataFrame(geometry=grid_polys, crs=target_crs)
grid_gdf["centroid"] = [Point(c) for c in grid_centroids]
grid_gdf = grid_gdf.set_geometry("geometry")

# Clip grid to near WFIGS bounding box / CALFIRE extents to reduce size
study_buffer = 5 * 1000  # 5 km buffer around WFIGS extents
wf_bounds = wf_gdf.total_bounds
study_box = box(wf_bounds[0]-study_buffer, wf_bounds[1]-study_buffer, wf_bounds[2]+study_buffer, wf_bounds[3]+study_buffer)
grid_gdf = grid_gdf[grid_gdf.intersects(study_box)].reset_index(drop=True)
print("Grid cells after clipping:", len(grid_gdf))

# Feature 1: area burned historically within the cell (sum of intersected perimeter areas)
# Ensure calfire polygons have a numeric area field; if not compute polygon intersection area
calfire_gdf["perim_area"] = calfire_gdf.geometry.area  # area in m^2

def compute_burned_area_for_cell(cell_geom):
    # intersection area between cell and all perimeters
    inter = calfire_gdf.geometry.intersection(cell_geom)
    # sum area of intersections
    return sum([g.area for g in inter if not g.is_empty])

burned_areas = []
# This loop can be heavy; we optimize a bit by spatial index
calfire_sindex = calfire_gdf.sindex
for geom in tqdm(grid_gdf.geometry, desc="computing burned area per cell"):
    possible_idx = list(calfire_sindex.intersection(geom.bounds))
    if len(possible_idx)==0:
        burned_areas.append(0.0)
        continue
    inter_area = 0.0
    for idx in possible_idx:
        poly = calfire_gdf.geometry.iloc[idx]
        if poly.intersects(geom):
            inter = poly.intersection(geom)
            if not inter.is_empty:
                inter_area += inter.area
    burned_areas.append(inter_area)

grid_gdf["burned_area_m2"] = burned_areas
grid_gdf["burned_area_km2"] = grid_gdf["burned_area_m2"] / 1e6

# Feature 2: local burn frequency (how many distinct perimeters intersect the cell)
def count_perimeters_in_cell(cell_geom):
    possible_idx = list(calfire_sindex.intersection(cell_geom.bounds))
    count = 0
    for idx in possible_idx:
        poly = calfire_gdf.geometry.iloc[idx]
        if poly.intersects(cell_geom):
            count += 1
    return count

burn_counts = []
for geom in tqdm(grid_gdf.geometry, desc="computing burn counts per cell"):
    burn_counts.append(count_perimeters_in_cell(geom))
grid_gdf["burn_count"] = burn_counts

# Feature 3: distance (meters) from the cell centroid to nearest historical perimeter boundary
# We'll compute distance to nearest perimeter geometry (zero if centroid inside any perimeter)
centroids = gpd.GeoSeries([c for c in grid_gdf.centroid], crs=target_crs)
# Use spatial index for speed
nearest_distances = []
for c in tqdm(centroids, desc="computing nearest distance to perimeters"):
    # quickly find candidates
    possible_idx = list(calfire_sindex.nearest(c.bounds, num_results=5))
    # compute exact min distance
    dmin = np.inf
    for idx in possible_idx:
        d = c.distance(calfire_gdf.geometry.iloc[idx])
        if d < dmin:
            dmin = d
    # if no candidate, set large distance
    if dmin == np.inf:
        dmin = max(maxx-minx, maxy-miny)
    nearest_distances.append(dmin)
grid_gdf["dist_to_perimeter_m"] = nearest_distances

# Label: whether any WFIGS incident point falls inside the cell -> positive class
# Build spatial index for wf_gdf
wf_sindex = wf_gdf.sindex
labels = []
for geom in tqdm(grid_gdf.geometry, desc="labeling cells with incidents"):
    possible = list(wf_sindex.intersection(geom.bounds))
    has_fire = False
    for idx in possible:
        p = wf_gdf.geometry.iloc[idx]
        if p.within(geom):
            has_fire = True
            break
    labels.append(1 if has_fire else 0)
grid_gdf["label_fire"] = labels

# Keep feature matrix
feature_cols = ["burned_area_km2", "burn_count", "dist_to_perimeter_m"]
X = grid_gdf[feature_cols].fillna(0).values
y = grid_gdf["label_fire"].values.astype(int)

# Normalize / transform features
# dist_to_perimeter: large distance -> likely lower risk; we'll invert it so larger -> more risk
# but better: scale all then let model learn weights
X_trans = X.copy()
# invert distance so proximity -> larger positive: prox_score = 1/(1 + dist_km)
X_trans[:,2] = 1.0 / (1.0 + (X_trans[:,2] / 1000.0))  # dist in km scaled to (0,1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_trans)

# Train/Test split (stratify to keep positives in both)
if y.sum() == 0:
    print("WARNING: no positive labels found in the grid (no incidents fell inside any grid cell).")
    # In that case, we can fallback to unsupervised scoring (e.g., normalized combination).
    # Create a risk score as normalized weighted sum:
    wa = 0.5  # weight for burned area
    wb = 0.3  # weight for burn count
    wc = 0.2  # weight for proximity
    raw_score = wa * (X_trans[:,0] / (X_trans[:,0].max() + 1e-9)) + \
                wb * (X_trans[:,1] / (X_trans[:,1].max() + 1e-9)) + \
                wc * (X_trans[:,2] / (X_trans[:,2].max() + 1e-9))
    # normalize to 0-1
    risk_score = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min() + 1e-9)
    grid_gdf["fire_likelihood"] = risk_score
    print("Produced unsupervised 0-1 risk score (no labels available).")
else:
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, grid_gdf.index.values, test_size=0.25, stratify=y, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_proba)
    print("Trained logistic regression. Test ROC AUC:", auc)
    # Apply model to all grid cells to get probability (0-1)
    all_proba = model.predict_proba(X_scaled)[:,1]
    grid_gdf["fire_likelihood"] = all_proba

# --- 4. Save results ---
out_csv = "/mnt/data/grid_fire_likelihood.csv"
grid_gdf.to_file("/mnt/data/grid_fire_likelihood.geojson", driver="GeoJSON")
# Save tabular CSV (centroid coords + score)
grid_out = grid_gdf.copy()
grid_out["centroid_x"] = [p.x for p in grid_out.centroid]
grid_out["centroid_y"] = [p.y for p in grid_out.centroid]
grid_out[["centroid_x", "centroid_y", "burned_area_km2", "burn_count", "dist_to_perimeter_m", "fire_likelihood"]].to_csv(out_csv, index=False)
print("Saved outputs:", out_csv, "/mnt/data/grid_fire_likelihood.geojson")

# --- 5. Quick visualization ---
fig, ax = plt.subplots(1,1, figsize=(10,10))
# plot base perimeters (light)
calfire_gdf.plot(ax=ax, linewidth=0.2, edgecolor="gray")
# plot grid cells colored by risk
grid_gdf.plot(column="fire_likelihood", ax=ax, alpha=0.8, legend=True)
wf_gdf.plot(ax=ax, markersize=5, color="black")
plt.title("Grid fire likelihood (0-1) — higher = greater predicted probability")
plt.axis("off")
plt.show()

# Print short summary stats
print("Risk score stats: min {:.3f}, mean {:.3f}, max {:.3f}".format(
    grid_gdf["fire_likelihood"].min(),
    grid_gdf["fire_likelihood"].mean(),
    grid_gdf["fire_likelihood"].max()
))
