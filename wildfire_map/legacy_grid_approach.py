"""
Legacy verbose grid generation approach - moved from main.py for reference.
This is the old way of doing grid-based wildfire risk analysis.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
from tqdm import tqdm


def build_grid_legacy(wf_gdf, cell_size=1000, study_buffer=5000):
    """
    Original verbose grid generation approach.
    This is kept for reference but replaced with cleaner alternatives.
    """
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


def compute_features_legacy(grid_gdf, calfire_gdf):
    """
    Original verbose feature computation approach.
    """
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


def label_cells_legacy(grid_gdf, wf_gdf):
    """
    Original verbose cell labeling approach.
    """
    wf_sindex = wf_gdf.sindex
    labels = []
    for geom in tqdm(grid_gdf.geometry, desc="labeling cells"):
        possible = list(wf_sindex.intersection(geom.bounds))
        labels.append(any(wf_gdf.geometry.iloc[i].within(geom) for i in possible))
    grid_gdf["label_fire"] = np.array(labels, dtype=int)
    return grid_gdf
