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
def build_grid(wf_gdf, cell_size=10000, study_buffer=10000):
    """
    Clean GeoPandas-based grid generation - much faster and cleaner than the legacy approach.
    """
    # Get study area bounds and expand by buffer
    bounds = wf_gdf.total_bounds  # [minx, miny, maxx, maxy]
    minx, miny, maxx, maxy = bounds
    minx -= study_buffer
    miny -= study_buffer  
    maxx += study_buffer
    maxy += study_buffer
    
    # Create grid using numpy meshgrid (much faster than nested loops)
    x_coords = np.arange(minx, maxx, cell_size)
    y_coords = np.arange(miny, maxy, cell_size)
    
    # Generate all grid cells at once using list comprehension
    grid_cells = [
        box(x, y, x + cell_size, y + cell_size)
        for x in x_coords
        for y in y_coords
    ]
    
    # Create GeoDataFrame
    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=wf_gdf.crs)
    grid_gdf["centroid"] = grid_gdf.geometry.centroid
    
    # Clip grid to California boundary to remove the yellow rectangle
    try:
        ca_boundary = gpd.read_file("./data/california_boundary.geojson")
        ca_boundary = ca_boundary.to_crs(grid_gdf.crs)
        ca_boundary = ca_boundary.dissolve()  # Combine all counties into single boundary
        
        # Keep only grid cells that intersect with California
        grid_gdf = grid_gdf[grid_gdf.intersects(ca_boundary.geometry.iloc[0])].reset_index(drop=True)
        print(f"Grid clipped to California boundary: {len(grid_gdf)} cells remain")
    except Exception as e:
        print(f"Could not clip to California boundary: {e}")
        print("Grid will show rectangular area - this may cause the yellow rectangle")
    
    print(f"Grid size: {len(x_coords)} × {len(y_coords)} = {len(grid_gdf)} cells")
    return grid_gdf

# --- 5. Compute features ---
def compute_features(grid_gdf, calfire_gdf):
    """
    More efficient feature computation using vectorized operations where possible.
    """
    calfire_gdf = calfire_gdf.copy()
    calfire_gdf["perim_area"] = calfire_gdf.geometry.area
    
    # Use spatial join for burned area and count (much faster than loops)
    print("Computing burned area and fire counts using spatial join...")
    burned_join = gpd.sjoin(grid_gdf, calfire_gdf, how='left', predicate='intersects')
    
    # Group by grid cell and compute features - handle the multi-level columns properly
    if len(burned_join) > 0:
        # Group by the left index (grid cells)
        grouped = burned_join.groupby(burned_join.index)
        
        # Compute features for each grid cell
        burned_area = grouped['perim_area'].sum()
        burn_count = grouped.size()
        
        # Create result dataframe
        features = grid_gdf.copy()
        features['burned_area_m2'] = features.index.map(burned_area).fillna(0)
        features['burn_count'] = features.index.map(burn_count).fillna(0)
    else:
        # No intersections found
        features = grid_gdf.copy()
        features['burned_area_m2'] = 0
        features['burn_count'] = 0
    
    features['burned_area_km2'] = features['burned_area_m2'] / 1e6
    
    # Distance to nearest perimeter (still need to compute this)
    print("Computing distances to nearest fire perimeters...")
    grid_centroids = features.geometry.centroid
    nearest_distances = []
    
    for centroid in tqdm(grid_centroids, desc="computing distance"):
        distances = calfire_gdf.geometry.distance(centroid)
        nearest_distances.append(distances.min())
    
    features['dist_to_perimeter_m'] = nearest_distances
    
    return features

# --- 6. Label fire occurrence ---
def label_cells(grid_gdf, wf_gdf):
    """
    More efficient cell labeling using spatial join.
    """
    print("Labeling cells with WFIGS fire points...")
    # Use spatial join to find which grid cells contain WFIGS points
    fire_join = gpd.sjoin(grid_gdf, wf_gdf, how='inner', predicate='contains')
    
    # Create binary label: 1 if cell contains any fire, 0 otherwise
    grid_gdf["label_fire"] = 0  # Initialize all to 0
    grid_gdf.loc[fire_join.index, "label_fire"] = 1  # Set intersecting cells to 1
    
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
    """
    Create a proper California wildfire risk map with boundaries, labels, and 0-100 scale.
    """
    # Scale fire likelihood to 0-100
    grid_gdf["risk_score"] = (grid_gdf["fire_likelihood"] * 100).round(1)
    
    # Create figure with proper styling
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Add California boundary (from the data loader or boundary file)
    try:
        from utils import CALI_DATA_LOADER
        ca_boundary = gpd.read_file("./data/california_boundary.geojson")
        ca_boundary = ca_boundary.to_crs(grid_gdf.crs)
        ca_boundary.plot(ax=ax, color='none', edgecolor='black', linewidth=2, alpha=0.8)
    except:
        print("Could not load California boundary - using CAL FIRE perimeters as boundary")
        calfire_gdf.plot(ax=ax, color='none', edgecolor='gray', linewidth=0.5, alpha=0.6)
    
    # Plot risk grid with proper styling
    grid_gdf.plot(
        column="risk_score", 
        ax=ax, 
        alpha=0.8, 
        legend=True, 
        cmap="YlOrRd",
        legend_kwds={
            'label': 'Fire Risk Score (0-100)',
            'orientation': 'vertical',
            'shrink': 0.8,
            'aspect': 30
        },
        edgecolor='white',
        linewidth=0.1
    )
    
    # Add WFIGS fire points if available
    candidate_cols = ["FinalAcres", "GISAcres", "IncidentAcres", "Acres", "DailyAcres"]
    severity_col = next((c for c in candidate_cols if c in wf_gdf.columns), None)
    
    if severity_col is not None:
        wf_gdf[severity_col] = pd.to_numeric(wf_gdf[severity_col], errors="coerce").fillna(0)
        wf_gdf[severity_col] = np.clip(wf_gdf[severity_col], 0, wf_gdf[severity_col].quantile(0.99))
        
        # Plot fire points with size based on severity
        wf_gdf.plot(
            ax=ax, 
            column=severity_col, 
            cmap="Reds", 
            markersize=20 + 0.01 * wf_gdf[severity_col], 
            alpha=0.9,
            edgecolor='darkred',
            linewidth=0.5,
            legend=True,
            legend_kwds={
                'label': f'Fire Severity ({severity_col})',
                'orientation': 'vertical',
                'shrink': 0.6,
                'aspect': 20
            }
        )
    else:
        # Plot fire points without severity coloring
        wf_gdf.plot(
            ax=ax, 
            color='red', 
            markersize=15, 
            alpha=0.8,
            edgecolor='darkred',
            linewidth=0.5,
            label='Fire Incidents'
        )
    
    # Add title and labels
    plt.title("🔥 California Wildfire Risk Assessment\nGrid-based Risk Score (0-100) with Historical Fire Data", 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add axis labels
    ax.set_xlabel("Longitude (EPSG:3310)", fontsize=12)
    ax.set_ylabel("Latitude (EPSG:3310)", fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add text box with statistics
    risk_stats = f"""Risk Statistics:
Min: {grid_gdf['risk_score'].min():.1f}
Mean: {grid_gdf['risk_score'].mean():.1f}
Max: {grid_gdf['risk_score'].max():.1f}
High Risk Cells (>70): {(grid_gdf['risk_score'] > 70).sum()}
Total Grid Cells: {len(grid_gdf)}"""
    
    ax.text(0.02, 0.98, risk_stats, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend for fire points if no severity column
    if severity_col is None:
        ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Set equal aspect ratio and tight layout
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n📊 Risk Assessment Summary:")
    print(f"   Grid cells analyzed: {len(grid_gdf):,}")
    print(f"   Risk score range: {grid_gdf['risk_score'].min():.1f} - {grid_gdf['risk_score'].max():.1f}")
    print(f"   High risk cells (>70): {(grid_gdf['risk_score'] > 70).sum():,} ({(grid_gdf['risk_score'] > 70).mean()*100:.1f}%)")
    print(f"   Fire incidents plotted: {len(wf_gdf):,}")


def main():
    wf_gdf, calfire_gdf = load_inputs()
    grid = build_grid(wf_gdf)
    features = compute_features(grid, calfire_gdf)
    features = label_cells(features, wf_gdf)
    features = score(features)
    visualize(features, calfire_gdf, wf_gdf)


if __name__ == "__main__":
    main()