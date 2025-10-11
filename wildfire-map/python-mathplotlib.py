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
# Removed geopy imports - now using GeoJSON boundary for California filtering

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

def is_in_california_geojson(latitude, longitude, california_gdf=None):
    """
    Checks if a given latitude and longitude pair is within California using GeoJSON boundary data.
    Much faster and more accurate than geopy reverse geocoding.
    """
    if california_gdf is None:
        # Load California boundary GeoJSON if not provided
        california_gdf = gpd.read_file("California_County_Boundary_view_layer_for_public_use_-4560719754002809188.geojson")
        # Dissolve all counties into a single California boundary
        california_gdf = california_gdf.dissolve()
    
    # Create a point from the coordinates (lat/lon)
    point = Point(longitude, latitude)
    
    # Create a temporary GeoDataFrame with the point in lat/lon
    point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs="EPSG:4326")
    
    # Convert the point to the same CRS as the California boundary
    point_gdf = point_gdf.to_crs(california_gdf.crs)
    
    # Check if the point is within California
    return california_gdf.geometry.iloc[0].contains(point_gdf.geometry.iloc[0])

# --- 2. Load WFIGS incidents CSV ---
print("Loading WFIGS incidents CSV...")
wf_df = pd.read_csv(WFIGS_CSV, low_memory=False)
print("Columns in WFIGS CSV:", wf_df.columns.tolist())

# Detect coordinate columns - check both lat/lon and x/y columns
lat_col, lon_col = None, None

# First try to find lat/lon columns
for c in ["Latitude", "LAT", "lat", "InitialLatitude"]:
    if c in wf_df.columns:
        lat_col = c
        break
for c in ["Longitude", "LON", "lon", "InitialLongitude"]:
    if c in wf_df.columns:
        lon_col = c
        break

# If lat/lon not found, try x/y columns (might be in projected coordinates)
if not lat_col or not lon_col:
    print("Lat/lon columns not found, checking x/y columns...")
    for c in ["Y", "POINT_Y", "y"]:
        if c in wf_df.columns:
            lat_col = c
            break
    for c in ["X", "POINT_X", "x"]:
        if c in wf_df.columns:
            lon_col = c
            break

if not lat_col or not lon_col:
    raise RuntimeError("No latitude/longitude columns found.")

# Drop missing coords and convert to GeoDataFrame
# Drop missing coords
print(f"Original WFIGS points: {len(wf_df)}")
wf_df = wf_df.dropna(subset=[lat_col, lon_col]).copy()
print(f"WFIGS points after dropping missing coordinates: {len(wf_df)}")

# Check coordinate ranges
print(f"Coordinate range - {lat_col}: {wf_df[lat_col].min()} to {wf_df[lat_col].max()}")
print(f"Coordinate range - {lon_col}: {wf_df[lon_col].min()} to {wf_df[lon_col].max()}")

# Check if coordinates are in lat/lon format or projected coordinates
is_latlon = (wf_df[lat_col].min() >= -90 and wf_df[lat_col].max() <= 90 and 
             wf_df[lon_col].min() >= -180 and wf_df[lon_col].max() <= 180)

# Also check if they might be scaled lat/lon (e.g., degrees * 1000)
is_scaled_latlon = (wf_df[lat_col].min() >= -90000 and wf_df[lat_col].max() <= 90000 and 
                    wf_df[lon_col].min() >= -180000 and wf_df[lon_col].max() <= 180000)

if is_scaled_latlon:
    print("Coordinates appear to be scaled lat/lon, trying different scales...")
    
    # Try different scaling factors
    scales = [1000, 10000, 100000, 1000000]
    best_scale = None
    
    for scale in scales:
        test_lats = wf_df[lat_col].head(100) / scale
        test_lons = wf_df[lon_col].head(100) / scale
        
        lat_range = (test_lats.min(), test_lats.max())
        lon_range = (test_lons.min(), test_lons.max())
        
        print(f"  Scale {scale}: lat {lat_range[0]:.3f} to {lat_range[1]:.3f}, lon {lon_range[0]:.3f} to {lon_range[1]:.3f}")
        
        # Check if this looks like reasonable US coordinates
        if (lat_range[0] >= 25 and lat_range[1] <= 50 and 
            lon_range[0] >= -125 and lon_range[1] <= -65):
            print(f"  [OK] Scale {scale} looks like reasonable US coordinates!")
            best_scale = scale
            break
    
    if best_scale is None:
        print("  [X] No scale factor produced reasonable US coordinates")
        # Use original coordinates as-is
        wf_df["lat_converted"] = wf_df[lat_col]
        wf_df["lon_converted"] = wf_df[lon_col]
    else:
        print(f"Using scale factor: {best_scale}")
        wf_df["lat_converted"] = wf_df[lat_col] / best_scale
        wf_df["lon_converted"] = wf_df[lon_col] / best_scale
    
    print(f"Converted coordinate range - lat: {wf_df['lat_converted'].min():.6f} to {wf_df['lat_converted'].max():.6f}")
    print(f"Converted coordinate range - lon: {wf_df['lon_converted'].min():.6f} to {wf_df['lon_converted'].max():.6f}")
    
    # Use converted coordinates for filtering
    lat_col = "lat_converted"
    lon_col = "lon_converted"
elif not is_latlon:
    print("Coordinates appear to be in projected coordinate system, trying different projections...")
    
    # Try different common coordinate systems
    possible_crs = [
        "EPSG:3857",  # Web Mercator
        "EPSG:3310",  # California Albers
        "EPSG:4269",  # NAD83 (might be lat/lon but with different datum)
        "EPSG:4326",  # WGS84 (in case the check was wrong)
    ]
    
    best_conversion = None
    best_crs = None
    
    for crs in possible_crs:
        try:
            print(f"Trying CRS: {crs}")
            temp_gdf = gpd.GeoDataFrame(
                wf_df.head(100),  # Test with first 100 points
                geometry=[Point(xy) for xy in zip(wf_df[lon_col].head(100), wf_df[lat_col].head(100))],
                crs=crs
            )
            
            # Convert to lat/lon (WGS84)
            temp_gdf = temp_gdf.to_crs("EPSG:4326")
            
            # Check if conversion makes sense
            converted_lats = [geom.y for geom in temp_gdf.geometry]
            converted_lons = [geom.x for geom in temp_gdf.geometry]
            
            lat_range = (min(converted_lats), max(converted_lats))
            lon_range = (min(converted_lons), max(converted_lons))
            
            print(f"  Lat range: {lat_range[0]:.6f} to {lat_range[1]:.6f}")
            print(f"  Lon range: {lon_range[0]:.6f} to {lon_range[1]:.6f}")
            
            # Check if this looks like reasonable US coordinates
            if (lat_range[0] >= 25 and lat_range[1] <= 50 and 
                lon_range[0] >= -125 and lon_range[1] <= -65):
                print(f"  [OK] This looks like reasonable US coordinates!")
                best_conversion = (converted_lats, converted_lons)
                best_crs = crs
                break
            else:
                print(f"  [X] Coordinates don't look like US lat/lon")
                
        except Exception as e:
            print(f"  [X] Failed with {crs}: {e}")
    
    if best_conversion is None:
        print("Could not determine correct coordinate system. Using original coordinates as-is.")
        print("Note: This may result in no points being identified as within California.")
    else:
        print(f"Using CRS: {best_crs}")
        # Apply conversion to all data
        temp_gdf = gpd.GeoDataFrame(
            wf_df,
            geometry=[Point(xy) for xy in zip(wf_df[lon_col], wf_df[lat_col])],
            crs=best_crs
        )
        temp_gdf = temp_gdf.to_crs("EPSG:4326")
        
        wf_df["lat_converted"] = [geom.y for geom in temp_gdf.geometry]
        wf_df["lon_converted"] = [geom.x for geom in temp_gdf.geometry]
        
        print(f"Converted coordinate range - lat: {wf_df['lat_converted'].min():.6f} to {wf_df['lat_converted'].max():.6f}")
        print(f"Converted coordinate range - lon: {wf_df['lon_converted'].min():.6f} to {wf_df['lon_converted'].max():.6f}")
        
        # Use converted coordinates for filtering
        lat_col = "lat_converted"
        lon_col = "lon_converted"
else:
    print("Coordinates are already in lat/lon format")

# Pre-filter using California bounding box for performance
print("Pre-filtering using California bounding box...")
# California bounding box coordinates (Northern, Western, Eastern, Southern)
CA_NORTH = 42.00
CA_WEST = -124.42
CA_EAST = -114.12
CA_SOUTH = 32.43

print(f"California bounding box: {CA_SOUTH}°N to {CA_NORTH}°N, {CA_WEST}°W to {CA_EAST}°W")

# Apply bounding box filter first (much faster than GeoJSON)
bbox_mask = (
    (wf_df[lat_col] >= CA_SOUTH) & (wf_df[lat_col] <= CA_NORTH) &
    (wf_df[lon_col] >= CA_WEST) & (wf_df[lon_col] <= CA_EAST)
)

print(f"Points within California bounding box: {bbox_mask.sum()} out of {len(bbox_mask)}")
wf_df_bbox = wf_df[bbox_mask].copy()

if len(wf_df_bbox) == 0:
    print("No points within California bounding box!")
    wf_df = wf_df_bbox
else:
    # Now apply precise GeoJSON boundary filter to the pre-filtered points
    print("Loading California boundary for precise filtering...")
    california_gdf = gpd.read_file("California_County_Boundary_view_layer_for_public_use_-4560719754002809188.geojson")
    california_gdf = california_gdf.dissolve()  # Combine all counties into single boundary

    print("Applying precise California boundary filter...")
    # Test a few coordinates first
    test_coords = [(lat, lon) for lat, lon in zip(wf_df_bbox[lat_col].head(5), wf_df_bbox[lon_col].head(5))]
    print("Testing first 5 coordinates:")
    for i, (lat, lon) in enumerate(test_coords):
        result = is_in_california_geojson(lat, lon, california_gdf)
        print(f"  {i+1}. ({lat}, {lon}) -> {'IN' if result else 'OUT'} California")

    # Apply GeoJSON filter to the bounding box filtered data
    precise_mask = [is_in_california_geojson(lat, lon, california_gdf) 
                   for lat, lon in zip(wf_df_bbox[lat_col], wf_df_bbox[lon_col])]
    print(f"Precise California filter results: {sum(precise_mask)} out of {len(precise_mask)} points are in California")
    
    wf_df = wf_df_bbox[precise_mask].copy()
    print(f"WFIGS points after precise California filter: {len(wf_df)}")

# Convert to GeoDataFrame (keep this)
wf_gdf = gpd.GeoDataFrame(
    wf_df,
    geometry=[Point(xy) for xy in zip(wf_df[lon_col], wf_df[lat_col])],
    crs="EPSG:4326"
)


# Remove invalid geometries
wf_gdf = wf_gdf[wf_gdf.geometry.notnull() & ~wf_gdf.geometry.is_empty]
if wf_gdf.empty:
    raise RuntimeError("No valid WFIGS points after cleaning.")

# Reproject to California Albers (meters)
wf_gdf = wf_gdf.to_crs("EPSG:3310")
print("Loaded WFIGS points:", len(wf_gdf))
print("Reprojected WFIGS CRS:", wf_gdf.crs)


# --- 3. Load CAL FIRE perimeters ---
print("Downloading/reading CAL FIRE perimeters (FRAP)...")
calfire_url = CALFIRE_GEOJSON_URLS[1]  # you can cycle URLs if needed
calfire_gdf = gpd.read_file(calfire_url)

# Clean invalid geometries
calfire_gdf = calfire_gdf[~calfire_gdf.geometry.is_empty]
calfire_gdf = calfire_gdf[calfire_gdf.geometry.notnull()]
calfire_gdf["geometry"] = calfire_gdf["geometry"].buffer(0)
calfire_gdf = calfire_gdf[calfire_gdf.is_valid]

# Reproject to California Albers
calfire_gdf = calfire_gdf.to_crs("EPSG:3310")
print("CAL FIRE perimeters loaded. Features:", len(calfire_gdf))
print("CALFIRE bounds:", calfire_gdf.total_bounds)

# --- 4. Generate grid for modeling ---
# Approximate California bounds in EPSG:3310 (meters)
cell_size = 1000  # 1 km grid cells
ca_minx, ca_miny = -2500000, -2500000
ca_maxx, ca_maxy = 1100000, 1300000

# Clip WFIGS points to California bounds
wf_gdf = wf_gdf.cx[ca_minx:ca_maxx, ca_miny:ca_maxy]
if wf_gdf.empty:
    raise RuntimeError("No WFIGS points within California bounds.")

# Compute grid bounds safely
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

# Clip grid to area around WFIGS points (study area)
study_buffer = 5000
wf_bounds = wf_gdf.total_bounds  # Get bounds from the filtered WFIGS data
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

# --- 6. Label fire occurrence ---
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

# --- Safety check: Ensure grid_gdf is not empty ---
if len(grid_gdf) == 0:
    raise ValueError("Grid generation failed — no cells created. Check bounding box or CRS mismatch.")

print(grid_gdf)
print(grid_gdf.keys())
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
"""
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

*/
"""
# --- 7. Output and visualize ---
out_geojson = "grid_fire_likelihood.geojson"
out_csv = "grid_fire_likelihood.csv"
grid_gdf.to_file(out_geojson, driver="GeoJSON")

grid_out = grid_gdf.copy()
grid_out["centroid_x"] = [p.x for p in grid_out.centroid]
grid_out["centroid_y"] = [p.y for p in grid_out.centroid]
grid_out[["centroid_x", "centroid_y", "burned_area_km2", "burn_count", "dist_to_perimeter_m", "fire_likelihood"]].to_csv(out_csv, index=False)
print("Saved:", out_csv, out_geojson)

# ✅ FIX: ensure coordinate systems match
wf_gdf = wf_gdf.to_crs(grid_gdf.crs)

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
