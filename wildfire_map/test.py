#!/usr/bin/env python
"""
fire_spread_ml.py
Predict 24-h burn probability for an ongoing California fire.
Inputs:  --lat 38.123 --lon -120.456
Outputs: risk.tif   +   risk_contours.geojson
"""
import argparse, json, os, tempfile, requests, tarfile
import numpy as np
import xarray as xr
import geopandas as gpd
from pyproj import Transformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from tqdm import tqdm

# ---------- CONFIG ---------- #
CELL_SIZE = 1000          # 1 km pixels
SCENE_PAD  = 15_000       # m buffer around fire
DAY_BACK   = 7            # how many days of weather to aggregate
MODEL_SEED = 42

# ---------- HELPERS ---------- #
def get_noaa_grid(lat, lon):
    """Download 7-day forecast grid for a point (temperature, rh, wind)."""
    # 1. discover the NOAA grid point
    meta = requests.get(f"https://api.weather.gov/points/{lat},{lon}").json()
    grid = meta['properties']
    wx_url = grid['forecastGridData']
    ds = xr.open_dataset(wx_url.replace('https','http')+'.nc', engine='netcdf4')
    # 2. subset last 7 days
    ds = ds.sel(time=slice(str(np.datetime64('today','D') - np.timedelta64(DAY_BACK,'D')),
                           str(np.datetime64('today','D'))))
    return ds[['temperature','maxWindSpeed','relativeHumidity']]

def terrain_fuel(lat, lon, size):
    """Grab 1-km elevation & NFDB fuel type from Living Atlas tiles (cached)."""
    # we use a pre-baked 1-km California DEM + NFDB reclassified raster
    # hosted on GitHub for reproducibility (~12 MB)
    url = "https://github.com/opengeos/ca-terrain-fuel/releases/download/v1.0/ca_dem_fuel_1km.tif"
    local = "ca_dem_fuel_1km.tif"
    if not os.path.exists(local):
        print("Downloading 1-km CA DEM + fuel raster …")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local,'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    with rasterio.open(local) as src:
        win = src.window(*src.window_bounds(src.index(lon, lat)))
        win = win.round_offsets().round_lengths()
        elev = src.read(1, window=win)
        fuel = src.read(2, window=win)
        trans = src.window_transform(win)
        bounds = src.bounds
    return elev, fuel, trans, bounds

def historic_burn(lat, lon, size):
    """Return 1-km binary mask: did any FRAP perimeter ever burn here?"""
    perims = gpd.read_file("https://opendata.arcgis.com/api/v3/datasets/0b711d3faaba4185996ecf4c2b30d3c5_0/downloads/data?format=geojson&spatialRefId=4326",
                           bbox=(lon-size/111e3, lat-size/111e3, lon+size/111e3, lat+size/111e3))
    if perims.empty: return None
    perims = perims.to_crs(3310)  # CA Albers
    mask = rasterio.features.geometry_mask(perims.geometry, out_shape=(size, size), transform=rasterio.transform.from_bounds(*perims.total_bounds, size, size), invert=True)
    return mask.astype(np.uint8)

def build_training_scene(lat, lon, half_size=25_000):
    """Build a labelled 50 km × 50 km scene somewhere else in CA to train on."""
    # pick a random centre inside CA
    np.random.seed(MODEL_SEED)
    train_lat = lat + np.random.uniform(-2,2)
    train_lon = lon + np.random.uniform(-2,2)
    size_m = half_size*2
    wx = get_noaa_grid(train_lat, train_lon)
    tmean = wx.temperature.mean('time').values
    wmean = wx.maxWindSpeed.mean('time').values
    rhmean = wx.relativeHumidity.mean('time').values
    elev, fuel, trans, _ = terrain_fuel(train_lat, train_lon, size_m)
    everburn = historic_burn(train_lat, train_lon, size_m) or np.zeros_like(elev)
    # fake “next-day burn” labels: highly simplified – we treat any pixel
    # within 500 m of a *new* 2022 perimeter as “burned”.
    perims22 = gpd.read_file("https://opendata.arcgis.com/api/v3/datasets/0b711d3faaba4185996ecf4c2b30d3c5_0/downloads/data?format=geojson&spatialRefId=4326",
                             bbox=(train_lon-0.5, train_lat-0.5, train_lon+0.5, train_lat+0.5))
    if not perims22.empty:
        perims22 = perims22[perims22.YEAR_==2022].to_crs(3310)
        burned = rasterio.features.geometry_mask(perims22.buffer(500), elev.shape,
                                                 rasterio.transform.from_bounds(*perims22.total_bounds, *elev.shape),
                                                 invert=True)
    else:
        burned = np.zeros_like(elev, dtype=bool)
    # stack features
    mask = (elev!=-9999) & (fuel>0)
    X = np.stack([elev[mask], fuel[mask], tmean[mask], wmean[mask], rhmean[mask], everburn[mask]], axis=1)
    y = burned[mask]
    return X, y

def train_model(lat, lon):
    """Train a small GBM on one random 50 km scene in CA."""
    print("Building training set …")
    X, y = build_training_scene(lat, lon)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=MODEL_SEED)
    clf.fit(Xs, y)
    print(f"Model trained on {len(y)} 1-km pixels, {y.sum()} burned.")
    return clf, scaler

def predict_scene(lat, lon, clf, scaler):
    """Predict 24-h burn probability for the requested fire vicinity."""
    print("Downloading weather & terrain …")
    wx = get_noaa_grid(lat, lon)
    tmean = wx.temperature.mean('time').values
    wmean = wx.maxWindSpeed.mean('time').values
    rhmean = wx.relativeHumidity.mean('time').values
    elev, fuel, trans, bounds = terrain_fuel(lat, lon, SCENE_PAD*2)
    everburn = historic_burn(lat, lon, SCENE_PAD*2) or np.zeros_like(elev)
    mask = (elev!=-9999) & (fuel>0)
    X = np.stack([elev[mask], fuel[mask], tmean[mask], wmean[mask], rhmean[mask], everburn[mask]], axis=1)
    Xs = scaler.transform(X)
    prob = clf.predict_proba(Xs)[:,1]
    # map back to 2-D
    proba = np.full(elev.shape, np.nan, dtype=np.float32)
    proba[mask] = prob.astype(np.float32)
    return proba, trans, bounds

def write_raster_and_contours(proba, trans, bounds, prefix='risk'):
    """Save GeoTIFF + GeoJSON contours for three risk levels."""
    # GeoTIFF
    transform = from_bounds(*bounds, proba.shape[1], proba.shape[0])
    with rasterio.open(f'{prefix}.tif','w',driver='GTiff',
                       height=proba.shape[0], width=proba.shape[1],
                       count=1, dtype='float32', crs='EPSG:3310',
                       transform=transform, nodata=np.nan) as dst:
        dst.write(proba, 1)
    # Contours
    levels = [0.3, 0.6, 0.9]
    contours = []
    for level in levels:
        cs = rasterio.features.shapes((proba>=level).astype(np.uint8), transform=transform)
        for geom, val in cs:
            if val:
                contours.append({'type':'Feature',
                                 'geometry':geom,
                                 'properties':{'risk':'High' if level==0.3 else 'VeryHigh' if level==0.6 else 'Extreme'}})
    with open(f'{prefix}_contours.geojson','w') as f:
        json.dump({'type':'FeatureCollection','features':contours}, f)
    print(f"Saved {prefix}.tif + {prefix}_contours.geojson")

# ---------- CLI ---------- #
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lat', type=float, required=True, help='Fire latitude')
    ap.add_argument('--lon', type=float, required=True, help='Fire longitude')
    args = ap.parse_args()

    clf, scaler = train_model(args.lat, args.lon)
    proba, trans, bounds = predict_scene(args.lat, args.lon, clf, scaler)
    write_raster_and_contours(proba, trans, bounds)