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
from tqdm import tqdm
from historic_dataloader import CurrentFireDataLoader
# Removed geopy imports - now using GeoJSON boundary for California filtering

# --- 1. Paths & data sources ---
#WFIGS_CSV = r"C:\Users\anujx\Downloads\WFIGS_Incident_Locations_-835549102613066266 (2).csv"
WFIGS_CSV = r"./data/WFIGS_Incident_Locations_Current_1704360007229377145.csv"
BOUNDARY_GEOJSON = "./data/california_boundary.geojson"

CALFIRE_GEOJSON_URLS = [
    "https://data.ca.gov/dataset/california-fire-perimeters-all/resource/dd5e4337-8679-4d64-bd90-b9df85ee6b58/download", 
    "https://services.arcgis.com/jIL9msH9OI208GCb/ArcGIS/rest/services/California_Fire_Perimeters_1878_2019/FeatureServer/1/query?where=1%3D1&outFields=*&f=geojson",
    "https://gis.data.cnra.ca.gov/arcgis/rest/services/CALFIRE-Forestry/california-fire-perimeters-all/FeatureServer/0/query?where=1%3D1&outFields=*&f=geojson"
]

def load_inputs():
    dataloader = CurrentFireDataLoader(
        wfigs_csv_path=WFIGS_CSV,
        fire_perimeter_path=CALFIRE_GEOJSON_URLS[1]
    )
    wf = dataloader.load_wfigs_data()
    cal = dataloader.load_fire_perimeters()
    return wf, cal

# --- 4. Generate grid for modeling ---
# Approximate California bounds in EPSG:3310 (meters)
def build_grid(wf_gdf, cell_size=1000, study_buffer=5000):
    ca_minx, ca_miny = -2500000, -2500000
    ca_maxx, ca_maxy = 1100000, 1300000
    wf_gdf = wf_gdf.cx[ca_minx:ca_maxx, ca_miny:ca_maxy]
    if wf_gdf.empty:
        raise RuntimeError("No WFIGS points within California bounds.")

    minx, miny, maxx, maxy = wf_gdf.total_bounds
    nx = int(np.ceil((maxx - minx) / cell_size))
    ny = int(np.ceil((maxy - miny) / cell_size))
    print(f"Grid size: {nx} × {ny}")

    grid_polys, grid_centroids = [], []
    for i in range(nx):
        for j in range(ny):
            x0, y0 = minx + i*cell_size, miny + j*cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            grid_polys.append(box(x0, y0, x1, y1))
            grid_centroids.append(((x0+x1)/2, (y0+y1)/2))

    grid_gdf = gpd.GeoDataFrame(geometry=grid_polys, crs="EPSG:3310")
    grid_gdf["centroid"] = [Point(c) for c in grid_centroids]

    wf_bounds = wf_gdf.total_bounds
    study_box = box(wf_bounds[0]-study_buffer, wf_bounds[1]-study_buffer,
                    wf_bounds[2]+study_buffer, wf_bounds[3]+study_buffer)
    grid_gdf = grid_gdf[grid_gdf.intersects(study_box)].reset_index(drop=True)
    print("Grid cells after clipping:", len(grid_gdf))
    return grid_gdf

# --- 5. Compute features ---
def compute_features(grid_gdf, calfire_gdf):
    calfire_gdf = calfire_gdf.copy()
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

    centroids = gpd.GeoSeries([p for p in grid_gdf.centroid], crs="EPSG:3310")
    nearest_distances = []
    for c in tqdm(centroids, desc="computing distance"):
        try:
            possible_idx = list(calfire_sindex.nearest(c.bounds))
        except TypeError:
            possible_idx = list(calfire_sindex.intersection(c.bounds))
        dmin = np.min([c.distance(calfire_gdf.geometry.iloc[i]) for i in possible_idx]) if possible_idx else np.inf
        nearest_distances.append(dmin if np.isfinite(dmin) else 0)

    grid_gdf["dist_to_perimeter_m"] = nearest_distances
    return grid_gdf

# --- 6. Label fire occurrence ---
def label_cells(grid_gdf, wf_gdf):
    wf_sindex = wf_gdf.sindex
    labels = []
    for geom in tqdm(grid_gdf.geometry, desc="labeling cells"):
        possible = list(wf_sindex.intersection(geom.bounds))
        labels.append(any(wf_gdf.geometry.iloc[i].within(geom) for i in possible))
    grid_gdf["label_fire"] = np.array(labels, dtype=int)
    return grid_gdf

# --- 7. Model or heuristic scoring ---
def score(grid_gdf):
    feature_cols = ["burned_area_km2", "burn_count", "dist_to_perimeter_m"]
    X = grid_gdf[feature_cols].fillna(0).values
    if len(grid_gdf) == 0:
        raise ValueError("Grid generation failed — no cells created. Check bounding box or CRS mismatch.")

    X[:, 2] = 1.0 / (1.0 + (X[:, 2] / 1000.0))
    y = grid_gdf["label_fire"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if y.sum() == 0:
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
    return grid_gdf

def visualize(grid_gdf, calfire_gdf, wf_gdf):
    fig, ax = plt.subplots(figsize=(10,10))
    calfire_gdf.plot(ax=ax, linewidth=0.2, edgecolor="gray")
    grid_gdf.plot(column="fire_likelihood", ax=ax, alpha=0.7, legend=True, cmap="YlOrRd")

    candidate_cols = ["FinalAcres", "GISAcres", "IncidentAcres", "Acres", "DailyAcres"]
    severity_col = next((c for c in candidate_cols if c in wf_gdf.columns), None)
    if severity_col is None:
        print("No acreage column found; skipping severity coloring.")
    else:
        wf_gdf[severity_col] = pd.to_numeric(wf_gdf[severity_col], errors="coerce").fillna(0)
        wf_gdf[severity_col] = np.clip(wf_gdf[severity_col], 0, wf_gdf[severity_col].quantile(0.99))
        wf_gdf.plot(ax=ax, column=severity_col, cmap="hot_r", markersize=6 + 0.001 * wf_gdf[severity_col], alpha=0.8, legend=True)

    plt.title("🔥 California Wildfires — Likelihood & Severity")
    plt.axis("off")
    plt.show()


def main():
    wf_gdf, calfire_gdf = load_inputs()
    grid = build_grid(wf_gdf)
    grid = compute_features(grid, calfire_gdf)
    grid = label_cells(grid, wf_gdf)
    grid = score(grid)
    visualize(grid, calfire_gdf, wf_gdf)


if __name__ == "__main__":
    main()