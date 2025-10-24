"""
Point-based wildfire risk analysis - no grid needed!
"""

import numpy as np
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def analyze_fire_risk_points(wfigs_gdf, calfire_gdf, sample_density=1000):
    """
    Analyze fire risk at sample points instead of a full grid.
    Much more efficient for large areas.
    """
    # Create sample points across California
    bounds = wfigs_gdf.total_bounds
    minx, miny, maxx, maxy = bounds
    
    # Generate sample points
    x_coords = np.linspace(minx, maxx, int((maxx - minx) / sample_density))
    y_coords = np.linspace(miny, maxy, int((maxy - miny) / sample_density))
    
    sample_points = []
    for x in x_coords:
        for y in y_coords:
            sample_points.append(gpd.points_from_xy([x], [y])[0])
    
    sample_gdf = gpd.GeoDataFrame(geometry=sample_points, crs=wfigs_gdf.crs)
    
    # Compute features for each sample point
    features = compute_point_features(sample_gdf, calfire_gdf, wfigs_gdf)
    
    return features


def compute_point_features(sample_gdf, calfire_gdf, wfigs_gdf):
    """
    Compute risk features for each sample point.
    """
    # Distance to nearest fire perimeter
    fire_centroids = np.array([[p.x, p.y] for p in calfire_gdf.geometry.centroid])
    sample_coords = np.array([[p.x, p.y] for p in sample_gdf.geometry])
    
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(fire_centroids)
    distances, _ = nn.kneighbors(sample_coords)
    
    sample_gdf['dist_to_fire'] = distances.flatten()
    
    # Count fires within radius
    fire_counts = []
    for point in sample_gdf.geometry:
        buffer = point.buffer(5000)  # 5km radius
        count = calfire_gdf.geometry.intersects(buffer).sum()
        fire_counts.append(count)
    
    sample_gdf['fire_count_5km'] = fire_counts
    
    # Label: is there a WFIGS point nearby?
    wfigs_coords = np.array([[p.x, p.y] for p in wfigs_gdf.geometry])
    nn_wfigs = NearestNeighbors(n_neighbors=1, radius=1000)  # 1km radius
    nn_wfigs.fit(wfigs_coords)
    
    labels = []
    for point in sample_gdf.geometry:
        point_coords = np.array([[point.x, point.y]])
        distances, _ = nn_wfigs.kneighbors(point_coords)
        labels.append(1 if distances[0][0] < 1000 else 0)
    
    sample_gdf['has_fire'] = labels
    
    return sample_gdf


def train_point_model(sample_gdf):
    """
    Train a model on the sample points.
    """
    feature_cols = ['dist_to_fire', 'fire_count_5km']
    X = sample_gdf[feature_cols].values
    y = sample_gdf['has_fire'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Predict risk
    risk_scores = model.predict_proba(X_scaled)[:, 1]
    sample_gdf['risk_score'] = risk_scores
    
    return sample_gdf, model, scaler


# This approach is much simpler and often more efficient!
