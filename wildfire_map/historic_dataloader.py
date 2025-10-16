import geopandas as gpd
from shapely.geometry import Point, box
from utils import CALI_DATA_LOADER
import pandas as pd
import pytest

class CurrentFireDataLoader:
    def __init__(self, wfigs_csv_path, fire_perimeter_path):
        self.wfigs_csv_path = wfigs_csv_path
        self.fire_perimeter_path = fire_perimeter_path
    
    def load_wfigs_data(self, to_crs=True):
        wfigs_df = pd.read_csv(self.wfigs_csv_path, low_memory=False)
        required_columns = {"InitialLatitude", "InitialLongitude", "x", "y"}
        if (not required_columns.issubset(wfigs_df.columns)):
            raise ValueError("""Code currently assumes WFIGS data has columns 'InitialLatitude' and 'InitialLongitude, "x", and "y".
                             Please ensure these columns are in you WFIGS CSV, or change the load function in the 
                             HistoricalFireDataLoader class to use a different CSV schema.""")
        
        # Check if most coordinates are in california
        num_points_to_check = min(200, len(wfigs_df))
        print("Filtering out points that are not in California...")
        wfigs_df = wfigs_df[wfigs_df.apply(lambda row: CALI_DATA_LOADER.is_in_california(latitude=row['y'], longitude=row['x']), axis=1)]
        wfigs_df = wfigs_df[["InitialLatitude", "InitialLongitude", "x", "y"]].dropna()
        wfigs_df = gpd.GeoDataFrame(
            wfigs_df,
            geometry=[Point(xy) for xy in zip(wfigs_df["x"], wfigs_df["y"])],
            crs="EPSG:4326"
        )
        # Remove invalid geometries
        wfigs_df = wfigs_df[wfigs_df.geometry.notnull() & ~wfigs_df.geometry.is_empty]
        if wfigs_df.empty:
            raise RuntimeError("No valid WFIGS points after cleaning.")
        if to_crs:
            # Reproject to California Albers (meters)
            wfigs_df = wfigs_df.to_crs("EPSG:3310")
        return wfigs_df
    
    def load_fire_perimeters(self):
        # --- 3. Load CAL FIRE perimeters ---
        calfire_gdf = gpd.read_file(self.fire_perimeter_path)

        # Clean invalid geometries
        calfire_gdf = calfire_gdf[~calfire_gdf.geometry.is_empty]
        calfire_gdf = calfire_gdf[calfire_gdf.geometry.notnull()]
        calfire_gdf["geometry"] = calfire_gdf["geometry"].buffer(0)
        calfire_gdf = calfire_gdf[calfire_gdf.is_valid]

        # Reproject to California Albers
        calfire_gdf = calfire_gdf.to_crs("EPSG:3310")
        print("CAL FIRE perimeters loaded. Features:", len(calfire_gdf))
        print("CALFIRE bounds:", calfire_gdf.total_bounds)
        return calfire_gdf



    
