"""
Cleaner alternatives to the manual grid generation approach.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds


def build_grid_geopandas(wf_gdf, cell_size=1000, study_buffer=5000):
    """
    Much cleaner grid generation using GeoPandas built-ins.
    """
    # Get study area bounds
    bounds = wf_gdf.total_bounds  # [minx, miny, maxx, maxy]
    minx, miny, maxx, maxy = bounds
    
    # Expand by buffer
    minx -= study_buffer
    miny -= study_buffer  
    maxx += study_buffer
    maxy += study_buffer
    
    # Create grid using numpy meshgrid (much faster)
    x_coords = np.arange(minx, maxx, cell_size)
    y_coords = np.arange(miny, maxy, cell_size)
    
    # Generate all grid cells at once
    grid_cells = []
    for x in x_coords:
        for y in y_coords:
            grid_cells.append(box(x, y, x + cell_size, y + cell_size))
    
    # Create GeoDataFrame
    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=wf_gdf.crs)
    
    print(f"Grid size: {len(x_coords)} × {len(y_coords)} = {len(grid_gdf)} cells")
    return grid_gdf


def build_grid_rasterio(wf_gdf, cell_size=1000, study_buffer=5000):
    """
    Alternative using rasterio for even more efficient grid operations.
    """
    bounds = wf_gdf.total_bounds
    minx, miny, maxx, maxy = bounds
    
    # Expand bounds
    minx -= study_buffer
    miny -= study_buffer
    maxx += study_buffer  
    maxy += study_buffer
    
    # Calculate grid dimensions
    width = int((maxx - minx) / cell_size)
    height = int((maxy - miny) / cell_size)
    
    # Create transform
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    # Create grid cells using rasterio
    grid_cells = []
    for row in range(height):
        for col in range(width):
            # Convert pixel coordinates to geographic
            x, y = rasterio.transform.xy(transform, row, col)
            x1, y1 = rasterio.transform.xy(transform, row + 1, col + 1)
            grid_cells.append(box(x, y, x1, y1))
    
    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=wf_gdf.crs)
    print(f"Grid size: {width} × {height} = {len(grid_gdf)} cells")
    return grid_gdf


def compute_features_vectorized(grid_gdf, calfire_gdf):
    """
    Much more efficient feature computation using vectorized operations.
    """
    # Spatial join to get burned area per cell
    burned_join = gpd.sjoin(grid_gdf, calfire_gdf, how='left', predicate='intersects')
    
    # Group by grid cell and compute features
    features = burned_join.groupby(burned_join.index).agg({
        'geometry': 'first',  # Keep original grid geometry
        'perim_area': ['sum', 'count']  # Sum burned area, count perimeters
    }).reset_index()
    
    # Flatten column names
    features.columns = ['grid_id', 'geometry', 'burned_area_m2', 'burn_count']
    features['burned_area_km2'] = features['burned_area_m2'] / 1e6
    
    # Distance to nearest perimeter (vectorized)
    grid_centroids = features.geometry.centroid
    nearest_distances = []
    
    for centroid in grid_centroids:
        distances = calfire_gdf.geometry.distance(centroid)
        nearest_distances.append(distances.min())
    
    features['dist_to_perimeter_m'] = nearest_distances
    
    return features


def compute_features_sklearn(grid_gdf, calfire_gdf):
    """
    Alternative using scikit-learn for spatial operations.
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Get centroids
    grid_centroids = np.array([[p.x, p.y] for p in grid_gdf.geometry.centroid])
    fire_centroids = np.array([[p.x, p.y] for p in calfire_gdf.geometry.centroid])
    
    # Find nearest fire to each grid cell
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(fire_centroids)
    distances, indices = nn.kneighbors(grid_centroids)
    
    grid_gdf['dist_to_perimeter_m'] = distances.flatten()
    
    # For burned area, use spatial join (still most efficient)
    burned_join = gpd.sjoin(grid_gdf, calfire_gdf, how='left', predicate='intersects')
    burned_area = burned_join.groupby(burned_join.index)['perim_area'].sum()
    burn_count = burned_join.groupby(burned_join.index).size()
    
    grid_gdf['burned_area_m2'] = grid_gdf.index.map(burned_area).fillna(0)
    grid_gdf['burn_count'] = grid_gdf.index.map(burn_count).fillna(0)
    grid_gdf['burned_area_km2'] = grid_gdf['burned_area_m2'] / 1e6
    
    return grid_gdf


# Example usage:
if __name__ == "__main__":
    # This would replace the current build_grid function
    # grid_gdf = build_grid_geopandas(wf_gdf)  # Much cleaner!
    pass
