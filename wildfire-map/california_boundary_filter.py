#!/usr/bin/env python
"""
California Boundary Filter
Uses the official California county boundary GeoJSON data to efficiently filter points
to only include those within California state boundaries.

Based on data from: https://data.ca.gov/dataset/california-county-boundary-public/resource/a150c41e-a189-472b-866c-0e129a14e0e2
"""

import requests
import json
import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.prepared import prep
import geopandas as gpd
from typing import Union, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class CaliforniaBoundaryFilter:
    """
    Efficient California boundary filter using official GeoJSON data.
    
    This class downloads and caches the California county boundary data,
    then provides fast point-in-polygon testing for filtering coordinates.
    """
    
    def __init__(self, cache_file='california_boundary.geojson'):
        """
        Initialize the California boundary filter.
        
        Args:
            cache_file (str): Local file to cache the GeoJSON data
        """
        self.cache_file = cache_file
        self.boundary_geometry = None
        self.prepared_boundary = None
        self._load_boundary_data()
    
    def _download_boundary_data(self):
        """Download California boundary GeoJSON from the official data source."""
        url = "https://gis.data.ca.gov/api/download/v1/items/2c4e32ee61824e28bd4cf37ecac96f39/geojson?layers=0"
        
        print("Downloading California boundary data...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(self.cache_file, 'w') as f:
                json.dump(response.json(), f)
            
            print(f"California boundary data saved to {self.cache_file}")
            return True
            
        except requests.RequestException as e:
            print(f"Error downloading boundary data: {e}")
            return False
    
    def _load_boundary_data(self):
        """Load California boundary data from cache or download if needed."""
        # Check if cached file exists
        if not os.path.exists(self.cache_file):
            if not self._download_boundary_data():
                raise RuntimeError("Failed to download California boundary data")
        
        # Load the GeoJSON data
        try:
            with open(self.cache_file, 'r') as f:
                geojson_data = json.load(f)
            
            # Convert to GeoDataFrame
            gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
            
            # Dissolve all counties into a single California boundary
            self.boundary_geometry = gdf.dissolve().geometry.iloc[0]
            
            # Create prepared geometry for faster point-in-polygon tests
            self.prepared_boundary = prep(self.boundary_geometry)
            
            print("California boundary data loaded successfully")
            
        except Exception as e:
            print(f"Error loading boundary data: {e}")
            # Try to re-download if loading failed
            if self._download_boundary_data():
                self._load_boundary_data()
            else:
                raise RuntimeError("Failed to load California boundary data")
    
    def is_in_california(self, latitude: float, longitude: float) -> bool:
        """
        Check if a single point is within California boundaries.
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            
        Returns:
            bool: True if point is within California, False otherwise
        """
        point = Point(longitude, latitude)
        return self.prepared_boundary.contains(point)
    
    def filter_points(self, latitudes: Union[List[float], np.ndarray], 
                     longitudes: Union[List[float], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter arrays of coordinates to only include points within California.
        
        Args:
            latitudes: Array or list of latitude values
            longitudes: Array or list of longitude values
            
        Returns:
            Tuple of (filtered_latitudes, filtered_longitudes) as numpy arrays
        """
        latitudes = np.asarray(latitudes)
        longitudes = np.asarray(longitudes)
        
        if len(latitudes) != len(longitudes):
            raise ValueError("Latitude and longitude arrays must have the same length")
        
        # Create mask for points within California
        mask = np.array([self.is_in_california(lat, lon) 
                        for lat, lon in zip(latitudes, longitudes)])
        
        return latitudes[mask], longitudes[mask]
    
    def filter_dataframe(self, df: pd.DataFrame, 
                        lat_col: str = 'lat', 
                        lon_col: str = 'lon') -> pd.DataFrame:
        """
        Filter a pandas DataFrame to only include rows with coordinates within California.
        
        Args:
            df: DataFrame containing coordinate columns
            lat_col: Name of the latitude column
            lon_col: Name of the longitude column
            
        Returns:
            Filtered DataFrame with only California points
        """
        if lat_col not in df.columns or lon_col not in df.columns:
            raise ValueError(f"DataFrame must contain columns '{lat_col}' and '{lon_col}'")
        
        # Create mask for points within California
        mask = df.apply(lambda row: self.is_in_california(row[lat_col], row[lon_col]), axis=1)
        
        return df[mask].copy()
    
    def get_california_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get the bounding box of California (min_lon, min_lat, max_lon, max_lat).
        
        Returns:
            Tuple of (min_longitude, min_latitude, max_longitude, max_latitude)
        """
        bounds = self.boundary_geometry.bounds
        return bounds[0], bounds[1], bounds[2], bounds[3]  # minx, miny, maxx, maxy


def is_in_california_geojson(latitude: float, longitude: float, 
                           filter_instance: CaliforniaBoundaryFilter = None) -> bool:
    """
    Standalone function to check if a point is within California using GeoJSON boundary.
    
    Args:
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
        filter_instance: Optional pre-initialized filter instance for efficiency
        
    Returns:
        bool: True if point is within California, False otherwise
    """
    if filter_instance is None:
        filter_instance = CaliforniaBoundaryFilter()
    
    return filter_instance.is_in_california(latitude, longitude)


def filter_california_points(latitudes: Union[List[float], np.ndarray], 
                           longitudes: Union[List[float], np.ndarray],
                           filter_instance: CaliforniaBoundaryFilter = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standalone function to filter coordinate arrays to only include California points.
    
    Args:
        latitudes: Array or list of latitude values
        longitudes: Array or list of longitude values
        filter_instance: Optional pre-initialized filter instance for efficiency
        
    Returns:
        Tuple of (filtered_latitudes, filtered_longitudes) as numpy arrays
    """
    if filter_instance is None:
        filter_instance = CaliforniaBoundaryFilter()
    
    return filter_instance.filter_points(latitudes, longitudes)


# Example usage and testing
if __name__ == "__main__":
    # Initialize the filter
    ca_filter = CaliforniaBoundaryFilter()
    
    # Test with some sample coordinates
    test_coords = [
        (37.7749, -122.4194),  # San Francisco, CA - should be True
        (34.0522, -118.2437),  # Los Angeles, CA - should be True
        (40.7128, -74.0060),   # New York, NY - should be False
        (51.5074, -0.1278),    # London, UK - should be False
        (36.7783, -119.4179),  # Fresno, CA - should be True
        (32.7157, -117.1611),  # San Diego, CA - should be True
    ]
    
    print("Testing California boundary filter:")
    print("-" * 50)
    
    for lat, lon in test_coords:
        is_in_ca = ca_filter.is_in_california(lat, lon)
        print(f"({lat:8.4f}, {lon:9.4f}) -> {'IN' if is_in_ca else 'OUT'} California")
    
    # Test with arrays
    print("\nTesting with coordinate arrays:")
    print("-" * 50)
    
    lats = np.array([lat for lat, lon in test_coords])
    lons = np.array([lon for lat, lon in test_coords])
    
    filtered_lats, filtered_lons = ca_filter.filter_points(lats, lons)
    
    print(f"Original points: {len(lats)}")
    print(f"California points: {len(filtered_lats)}")
    print("Filtered coordinates:")
    for lat, lon in zip(filtered_lats, filtered_lons):
        print(f"  ({lat:8.4f}, {lon:9.4f})")
    
    # Get California bounds
    bounds = ca_filter.get_california_bounds()
    print(f"\nCalifornia bounds: {bounds}")
