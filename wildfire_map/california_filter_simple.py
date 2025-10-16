#!/usr/bin/env python
"""
Simple California Boundary Filter
Uses the official California county boundary GeoJSON data to filter points
to only include those within California state boundaries.

This version uses only standard library modules and requests.
Based on data from: https://data.ca.gov/dataset/california-county-boundary-public/resource/a150c41e-a189-472b-866c-0e129a14e0e2
"""

import requests
import json
import os
import math
from typing import Union, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SimpleCaliforniaFilter:
    """
    Simple California boundary filter using official GeoJSON data.
    
    This class downloads and caches the California county boundary data,
    then provides point-in-polygon testing using the ray casting algorithm.
    """
    
    def __init__(self, cache_file='california_boundary.geojson'):
        """
        Initialize the California boundary filter.
        
        Args:
            cache_file (str): Local file to cache the GeoJSON data
        """
        self.cache_file = cache_file
        self.boundary_polygons = []
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
            
            # Extract all polygons from the GeoJSON
            self.boundary_polygons = []
            for feature in geojson_data['features']:
                geometry = feature['geometry']
                if geometry['type'] == 'Polygon':
                    # Convert coordinates to simple list of (lon, lat) tuples
                    coords = geometry['coordinates'][0]  # Exterior ring
                    polygon = [(coord[0], coord[1]) for coord in coords]
                    self.boundary_polygons.append(polygon)
                elif geometry['type'] == 'MultiPolygon':
                    # Handle MultiPolygon by adding each polygon
                    for polygon_coords in geometry['coordinates']:
                        coords = polygon_coords[0]  # Exterior ring
                        polygon = [(coord[0], coord[1]) for coord in coords]
                        self.boundary_polygons.append(polygon)
            
            # Check if coordinates are in projected system (large numbers) and convert to lat/lon
            if self.boundary_polygons:
                sample_coord = self.boundary_polygons[0][0]
                if abs(sample_coord[0]) > 180 or abs(sample_coord[1]) > 90:
                    print("Converting projected coordinates to lat/lon...")
                    self._convert_to_latlon()
            
            print(f"California boundary data loaded successfully ({len(self.boundary_polygons)} polygons)")
            
        except Exception as e:
            print(f"Error loading boundary data: {e}")
            # Try to re-download if loading failed
            if self._download_boundary_data():
                self._load_boundary_data()
            else:
                raise RuntimeError("Failed to load California boundary data")
    
    def _convert_to_latlon(self):
        """Convert projected coordinates to lat/lon using simple approximation."""
        # This is a simplified conversion - for production use, consider using pyproj
        converted_polygons = []
        
        for polygon in self.boundary_polygons:
            converted_polygon = []
            for x, y in polygon:
                # Simple Web Mercator to lat/lon conversion
                # This is approximate - for better accuracy use proper projection libraries
                lon = x / 20037508.34 * 180
                lat = math.atan(math.sinh(math.pi * (1 - 2 * y / 20037508.34))) * 180 / math.pi
                converted_polygon.append((lon, lat))
            converted_polygons.append(converted_polygon)
        
        self.boundary_polygons = converted_polygons
    
    def _point_in_polygon(self, x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
        """
        Check if a point is inside a polygon using the ray casting algorithm.
        
        Args:
            x: Longitude of the point
            y: Latitude of the point
            polygon: List of (longitude, latitude) tuples defining the polygon
            
        Returns:
            bool: True if point is inside polygon, False otherwise
        """
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def is_in_california(self, latitude: float, longitude: float) -> bool:
        """
        Check if a single point is within California boundaries.
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            
        Returns:
            bool: True if point is within California, False otherwise
        """
        # Check if point is in any of the California polygons
        for polygon in self.boundary_polygons:
            if self._point_in_polygon(longitude, latitude, polygon):
                return True
        return False
    
    def filter_points(self, latitudes: Union[List[float], list], 
                     longitudes: Union[List[float], list]) -> Tuple[list, list]:
        """
        Filter arrays of coordinates to only include points within California.
        
        Args:
            latitudes: List of latitude values
            longitudes: List of longitude values
            
        Returns:
            Tuple of (filtered_latitudes, filtered_longitudes) as lists
        """
        if len(latitudes) != len(longitudes):
            raise ValueError("Latitude and longitude arrays must have the same length")
        
        filtered_lats = []
        filtered_lons = []
        
        for lat, lon in zip(latitudes, longitudes):
            if self.is_in_california(lat, lon):
                filtered_lats.append(lat)
                filtered_lons.append(lon)
        
        return filtered_lats, filtered_lons
    
    def get_california_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get the bounding box of California (min_lon, min_lat, max_lon, max_lat).
        
        Returns:
            Tuple of (min_longitude, min_latitude, max_longitude, max_latitude)
        """
        all_lons = []
        all_lats = []
        
        for polygon in self.boundary_polygons:
            for lon, lat in polygon:
                all_lons.append(lon)
                all_lats.append(lat)
        
        return min(all_lons), min(all_lats), max(all_lons), max(all_lats)


def is_in_california_geojson(latitude: float, longitude: float, 
                           filter_instance: SimpleCaliforniaFilter = None) -> bool:
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
        filter_instance = SimpleCaliforniaFilter()
    
    return filter_instance.is_in_california(latitude, longitude)


def filter_california_points(latitudes: Union[List[float], list], 
                           longitudes: Union[List[float], list],
                           filter_instance: SimpleCaliforniaFilter = None) -> Tuple[list, list]:
    """
    Standalone function to filter coordinate arrays to only include California points.
    
    Args:
        latitudes: List of latitude values
        longitudes: List of longitude values
        filter_instance: Optional pre-initialized filter instance for efficiency
        
    Returns:
        Tuple of (filtered_latitudes, filtered_longitudes) as lists
    """
    if filter_instance is None:
        filter_instance = SimpleCaliforniaFilter()
    
    return filter_instance.filter_points(latitudes, longitudes)


# Example usage and testing
if __name__ == "__main__":
    # Initialize the filter
    ca_filter = SimpleCaliforniaFilter()
    
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
    
    lats = [lat for lat, lon in test_coords]
    lons = [lon for lat, lon in test_coords]
    
    filtered_lats, filtered_lons = ca_filter.filter_points(lats, lons)
    
    print(f"Original points: {len(lats)}")
    print(f"California points: {len(filtered_lats)}")
    print("Filtered coordinates:")
    for lat, lon in zip(filtered_lats, filtered_lons):
        print(f"  ({lat:8.4f}, {lon:9.4f})")
    
    # Get California bounds
    bounds = ca_filter.get_california_bounds()
    print(f"\nCalifornia bounds: {bounds}")
