#!/usr/bin/env python
"""
ca_wildfire_risk.py  –  end-to-end pipeline
1. Train / validate GBM on open CA data.
2. Scan state tile-by-tile.
3. Export CSV: lat,lon,prob  for highest-risk pixels.
---------------------------------------------------------
Usage:
    python ca_wildfire_risk.py          # full state scan
    python ca_wildfire_risk.py --help   # see options
"""
import os 
import json

import argparse 
import tempfile
import tarfile 
import requests
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pyproj import Transformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, precision_recall_fscore_support,
                             roc_curve, brier_score_loss)
import rasterio, rasterio.features
from rasterio.transform import from_bounds
from tqdm import tqdm
import joblib
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve

# ----------- CONFIG ---------- #
CELL         = 1_000     # 1 km
THRESHOLD    = 0.8       # risk cut-off for CSV
N_EST        = 300
MAX_DEPTH    = 4
MODEL_SEED   = 42
CA_W, CA_S, CA_E, CA_N = -124.5, 32.25, -114.0, 42.0  # state bbox
# ----------------------------- #

def get_noaa_grid(lat, lon):
    """Fetch 7-day weather aggregates (JSON API) and return xarray.Dataset-like object."""
    meta = requests.get(f"https://api.weather.gov/points/{lat},{lon}", timeout=30).json()
    wx_url = meta['properties']['forecastGridData']
    print("Fetching NOAA weather data …")

    resp = requests.get(wx_url, headers={"User-Agent": "wildfire-risk-app"}, timeout=30)
    resp.raise_for_status()
    grid = resp.json()

    # Extract time-series (simplified: temperature, wind, humidity)
    temp_vals = [v["value"] for v in grid["properties"]["temperature"]["values"] if v["value"] is not None]
    wind_vals = [v["value"] for v in grid["properties"]["windSpeed"]["values"] if v["value"] is not None]
    rh_vals   = [v["value"] for v in grid["properties"]["relativeHumidity"]["values"] if v["value"] is not None]

    times = [v["validTime"].split("/")[0] for v in grid["properties"]["temperature"]["values"][:len(temp_vals)]]

    # Construct xarray.Dataset manually
    ds = xr.Dataset(
        {
            "temperature": ("time", temp_vals),
            "maxWindSpeed": ("time", wind_vals),
            "relativeHumidity": ("time", rh_vals),
        },
        coords={"time": np.array(times, dtype="datetime64")}
    )
    return ds


def terrain_fuel_tile(left, bottom, right, top, width_px, height_px):
    """Download 1-km CA DEM + fuel raster tile; returns elev, fuel, transform, crs."""
    url = "https://github.com/opengeos/ca-terrain-fuel/releases/download/v1.0/ca_dem_fuel_1km.tif"
    local = "ca_dem_fuel_1km.tif"
    if not os.path.exists(local):
        print("Fetching 1-km CA DEM + fuel raster …")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local,'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    with rasterio.open(local) as src:
        win = src.window(left, bottom, right, top)
        win = win.round_offsets().round_lengths()
        elev = src.read(1, window=win)
        fuel = src.read(2, window=win)
        new_transform = src.window_transform(win)
    return elev, fuel, new_transform, src.crs

def historic_burn_mask(left, bottom, right, top, shape):
    """Binary mask 1=ever burned 1950-2022."""
    perims = gpd.read_file(
        "https://opendata.arcgis.com/api/v3/datasets/0b711d3faaba4185996ecf4c2b30d3c5_0/downloads/data?format=geojson&spatialRefId=4326",
        bbox=(left, bottom, right, top))
    if perims.empty:
        return np.zeros(shape, dtype=np.uint8)
    perims = perims.to_crs(3310)
    mask = rasterio.features.geometry_mask(
        perims.buffer(500), shape,
        rasterio.transform.from_bounds(*perims.total_bounds, *shape[::-1]),
        invert=True)
    return mask.astype(np.uint8)

def build_training_scene(lat, lon, half_side=50_000):
    """Random 100 km × 100 km labelled scene."""
    rng = np.random.default_rng(MODEL_SEED)
    train_lat = lat + rng.uniform(-1.5, 1.5)
    train_lon = lon + rng.uniform(-1.5, 1.5)
    size_m = half_side*2
    wx = get_noaa_grid(train_lat, train_lon)
    tmean = wx.temperature.mean('time').values
    wmean = wx.maxWindSpeed.mean('time').values
    rhmean = wx.relativeHumidity.mean('time').values
    elev, fuel, _, _ = terrain_fuel_tile(train_lon-1, train_lat-1, train_lon+1, train_lat+1, 100, 100)
    everburn = historic_burn_mask(train_lon-1, train_lat-1, train_lon+1, train_lat+1, elev.shape)
    # labels: 2022 perims
    perims22 = gpd.read_file(
        "https://opendata.arcgis.com/api/v3/datasets/0b711d3faaba4185996ecf4c2b30d3c5_0/downloads/data?format=geojson&spatialRefId=4326",
        bbox=(train_lon-0.8, train_lat-0.8, train_lon+0.8, train_lat+0.8))
    burned = np.zeros_like(elev, dtype=bool)
    if not perims22.empty:
        perims22 = perims22[perims22.YEAR_==2022].to_crs(3310)
        burned = rasterio.features.geometry_mask(
            perims22.buffer(500), elev.shape,
            rasterio.transform.from_bounds(*perims22.total_bounds, *elev.shape[::-1]),
            invert=True)
    mask = (elev!=-9999) & (fuel>0)
    X = np.stack([elev[mask], fuel[mask], tmean[mask], wmean[mask], rhmean[mask], everburn[mask]], axis=1)
    y = burned[mask]
    return X, y

def train_model(lat, lon):
    """Multi-tile train + validate; return clf, scaler, stats."""
    print("Collecting train/validation tiles …")
    X_list, y_list = [], []
    rng = np.random.default_rng(MODEL_SEED)
    for _ in range(5):
        la = rng.uniform(32.5, 42.0); lo = rng.uniform(-124.0, -114.0)
        X, y = build_training_scene(la, lo)
        X_list.append(X); y_list.append(y)

    X = np.vstack(X_list); y = np.hstack(y_list)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=MODEL_SEED, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    clf = GradientBoostingClassifier(
        n_estimators=N_EST,
        max_depth=MAX_DEPTH,
        learning_rate=0.05,
        random_state=MODEL_SEED
    )
    clf.fit(X_train_s, y_train)

    preds = clf.predict_proba(X_val_s)[:, 1]

    # Metrics
    auc   = roc_auc_score(y_val, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_val, preds >= 0.5, average='binary'
    )
    brier = brier_score_loss(y_val, preds)

    stats = dict(
        AUC=round(auc,3), F1=round(f1,3),
        Precision=round(prec,3), Recall=round(rec,3),
        Brier=round(brier,3)
    )
    print("Validation metrics:", stats)

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_val, preds)
    plt.figure()
    plt.plot(fpr, tpr, label=f"GBM (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png", dpi=120)
    plt.close()

    # --- Calibration Curve ---
    prob_true, prob_pred = calibration_curve(y_val, preds, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label="GBM")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend(loc="best")
    plt.savefig("calibration.png", dpi=120)
    plt.close()

    return clf, scaler, stats
def predict_tile(clf, scaler, left, bottom, right, top, width, height):
    """Return 2-D prob array for bbox tile."""
    wx = get_noaa_grid((bottom+top)/2, (left+right)/2)
    tmean = wx.temperature.mean('time').values
    wmean = wx.maxWindSpeed.mean('time').values
    rhmean = wx.relativeHumidity.mean('time').values
    elev, fuel, transform, crs = terrain_fuel_tile(left, bottom, right, top, width, height)
    everburn = historic_burn_mask(left, bottom, right, top, elev.shape)
    mask = (elev!=-9999) & (fuel>0)
    if mask.sum()==0:
        return np.full(elev.shape, np.nan, dtype=np.float32), transform, crs
    X = np.stack([elev[mask], fuel[mask], tmean[mask], wmean[mask], rhmean[mask], everburn[mask]], axis=1)
    Xs = scaler.transform(X)
    prob = clf.predict_proba(Xs)[:,1]
    proba = np.full(elev.shape, np.nan, dtype=np.float32)
    proba[mask] = prob.astype(np.float32)
    return proba, transform, crs

def tile_generator(step=0.5):
    """Cover CA with step×step degree tiles."""
    lons = np.arange(CA_W, CA_E, step)
    lats = np.arange(CA_S, CA_N, step)
    for lat in lats:
        for lon in lons:
            yield (lon, lat, lon+step, lat+step)

def ca_risk_scan(clf, scaler, outfile='high_risk_coords.csv'):
    """Scan CA, keep prob>=THRESHOLD, write CSV."""
    records=[]
    for left,bottom,right,top in tqdm(list(tile_generator()), desc='Scanning CA'):
        width  = int((right -left)*111e3 / CELL)
        height = int((top   -bottom)*111e3 / CELL)
        proba, transform, crs = predict_tile(clf, scaler, left, bottom, right, top, width, height)
        rows, cols = np.where(proba >= THRESHOLD)
        if len(rows)==0: continue
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        transformer = Transformer.from_crs(crs, 4326, always_xy=True)
        lons, lats = transformer.transform(xs, ys)
        for la, lo, pr in zip(lats, lons, proba[rows, cols]):
            records.append({'lat':round(float(la),5), 'lon':round(float(lo),5), 'prob':round(float(pr),3)})
    df = pd.DataFrame(records)
    df.to_csv(outfile, index=False)
    print(f"Done → {outfile}  ({len(df)} pixels)")

# ---------------- CLI ---------------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CA 24-h wildfire-risk scanner")
    parser.add_argument('--out', default='high_risk_coords.csv', help='output CSV')
    parser.add_argument('--threshold', type=float, default=THRESHOLD, help='prob cut-off')
    args = parser.parse_args()
    THRESHOLD = args.threshold
    clf, scaler, stats = train_model(37.0, -119.0)
    print("\nValidation summary:", stats)
    ca_risk_scan(clf, scaler, args.out)