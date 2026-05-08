"""
Full data pipeline: download 2024+2025+2026 data, merge, engineer features.

Run from the frigg_forecasting directory:
    python prepare_full_dataset.py

Replaces the processed and engineered CSV files with the extended dataset.
Then re-run notebook 04 only to retrain with ~2.5 years of data.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from src.data.loaders import EnergyChartsLoader, WeatherDataLoader
from src.data.fuel_loader import FuelPriceLoader
from src.features.engineer import FeatureEngineer

ZONES      = ['DE-LU', 'ES']
START_DATE = '2024-01-01'
END_DATE   = '2026-05-06'

energy_loader  = EnergyChartsLoader(cache_dir='data/raw')
weather_loader = WeatherDataLoader(cache_dir='data/external')
fuel_loader    = FuelPriceLoader(cache_dir='data/external')

print("\n[0/4] Downloading fuel prices (TTF gas, EUA carbon, Brent)...")
fuel_hourly = fuel_loader.load_fuel_prices(START_DATE, END_DATE, use_cache=True, resample_to_hourly=True)
fuel_hourly['timestamp'] = pd.to_datetime(fuel_hourly['timestamp'], utc=True)
fuel_hourly = fuel_hourly.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
print(f"      {len(fuel_hourly):,} hourly fuel price rows | cols: {[c for c in fuel_hourly.columns if c != 'timestamp']}")

Path('data/processed').mkdir(parents=True, exist_ok=True)
Path('data/engineered').mkdir(parents=True, exist_ok=True)

TIMEZONE = {'DE-LU': 'Europe/Berlin', 'ES': 'Europe/Madrid'}

for zone in ZONES:
    print(f"\n{'='*65}")
    print(f"  {zone}  |  {START_DATE} → {END_DATE}")
    print(f"{'='*65}")

    # ── 1. Download ──────────────────────────────────────────────────────────
    print("\n[1/4] Downloading prices...")
    prices = energy_loader.load_day_ahead_prices(zone, START_DATE, END_DATE, use_cache=True)
    prices['timestamp'] = pd.to_datetime(prices['timestamp'], utc=True)
    prices = prices.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    print(f"      {len(prices):,} rows")

    print("\n[2/4] Downloading generation data...")
    gen = energy_loader.load_generation_data(zone, START_DATE, END_DATE, use_cache=True)
    gen['timestamp'] = pd.to_datetime(gen['timestamp'], utc=True)
    gen = gen.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    # Resample to hourly if needed
    if gen['timestamp'].diff().median() < pd.Timedelta('1h'):
        gen = gen.set_index('timestamp').resample('h').mean().reset_index()
    print(f"      {len(gen):,} hourly rows  |  {list(gen.columns)}")

    print("\n[3/4] Downloading weather data...")
    weather = weather_loader.load_weather_data(zone, START_DATE, END_DATE, use_cache=True)
    weather['timestamp'] = pd.to_datetime(weather['timestamp'], utc=True)
    weather = weather.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    print(f"      {len(weather):,} rows")

    # ── 2. Merge ─────────────────────────────────────────────────────────────
    print("\n[4/4] Merging and engineering features...")
    merged = prices.merge(gen,        on='timestamp', how='inner')
    merged = merged.merge(weather,    on='timestamp', how='left')
    merged = merged.merge(fuel_hourly, on='timestamp', how='left')
    # Forward-fill fuel prices over any hourly gaps (weekends/holidays have no new price)
    fuel_cols = [c for c in fuel_hourly.columns if c != 'timestamp']
    merged[fuel_cols] = merged[fuel_cols].ffill().bfill()
    merged = merged.sort_values('timestamp').reset_index(drop=True)
    print(f"      Merged: {len(merged):,} rows, {len(merged.columns)} columns")

    # Save processed (raw merged) data
    proc_path = Path(f'data/processed/{zone.lower()}_merged_with_features.csv')
    df_save = merged.copy()
    df_save['timestamp'] = df_save['timestamp'].dt.tz_convert(TIMEZONE[zone]).apply(lambda x: x.isoformat())
    df_save.to_csv(proc_path, index=False)
    print(f"      Saved processed → {proc_path}")

    # ── 3. Feature Engineering ───────────────────────────────────────────────
    engineer = FeatureEngineer(zone=zone)
    df_feat  = engineer.fit_transform(merged)
    print(f"      Features engineered: {len(df_feat.columns)-1}")

    # Drop rows with NaN lag values (warm-up period)
    target_col   = 'price_eur_mwh'
    feature_cols = [c for c in df_feat.columns if c not in ['timestamp', target_col]]
    valid = ~(df_feat[feature_cols].isna().any(axis=1) | df_feat[target_col].isna())
    df_feat = df_feat[valid].reset_index(drop=True)
    print(f"      After NaN drop: {len(df_feat):,} samples")

    # Save engineered features
    feat_path = Path(f'data/engineered/{zone.lower()}_features.csv')
    df_feat_save = df_feat.copy()
    df_feat_save['timestamp'] = df_feat_save['timestamp'].dt.tz_convert(TIMEZONE[zone]).apply(lambda x: x.isoformat())
    df_feat_save.to_csv(feat_path, index=False)
    print(f"      Saved engineered → {feat_path}")

    # Save training data (X + y combined)
    train_path = Path(f'data/engineered/{zone.lower()}_training_data.csv')
    df_feat_save.drop(columns=['timestamp']).to_csv(train_path, index=False)
    print(f"      Saved training   → {train_path}")

print("\n" + "="*65)
print("  DONE — now re-run notebook 04 to train with the full dataset")
print("="*65)
