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