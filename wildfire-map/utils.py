import geopandas as gpd
from shapely.geometry import Point, box

class CaliforniaGeoDataLoader:
    def __init__(self, cache_file_path='./data/california_boundary.geojson'):
        self.cache_file_path = cache_file_path
        self.california_gdf = gpd.read_file(self.cache_file_path).dissolve()
    
    def is_in_california(self, latitude, longitude):

        # Create a point from the coordinates (lat/lon)
        point = Point(longitude, latitude)
        
        # Create a temporary GeoDataFrame with the point in lat/lon
        point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs="EPSG:4326")
        
        # Convert the point to the same CRS as the California boundary
        point_gdf = point_gdf.to_crs(self.california_gdf.crs)
        
        # Check if the point is within California
        return self.california_gdf.geometry.iloc[0].contains(point_gdf.geometry.iloc[0])

CALI_DATA_LOADER = CaliforniaGeoDataLoader()