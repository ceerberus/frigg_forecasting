"""
Full data pipeline: download 2024+2025+2026 data, merge, engineer features.

Run from the frigg_forecasting directory:
    python prepare_full_dataset.py

Optional RTE nuclear forecast (strongly recommended for DE-LU):
    Register free at https://digital.iservices.rte-france.com/
    set RTE_CLIENT_ID=...      (Windows)
    set RTE_CLIENT_SECRET=...
  Adds French nuclear D-1 generation forecasts — the key driver of DE-LU prices.

Optional ENTSO-E integration (cross-border flows, nuclear, hydro, forecasts):
    set ENTSOE_API_KEY=your_token   (Windows)
    export ENTSOE_API_KEY=your_token  (Mac/Linux)
  Then the script automatically adds ENTSO-E features.
  Without the key, the pipeline runs without ENTSO-E data.

Replaces the processed and engineered CSV files with the extended dataset.
Then re-run notebook 04 only to retrain with ~2.5 years of data.
"""

import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from src.data.loaders import EnergyChartsLoader, WeatherDataLoader
from src.data.fuel_loader import FuelPriceLoader
from src.data.neighbor_loader import NeighborGenerationLoader
from src.features.engineer import FeatureEngineer

ZONES      = ['DE-LU', 'ES']
START_DATE = '2024-01-01'
END_DATE   = '2026-05-08'

energy_loader   = EnergyChartsLoader(cache_dir='data/raw')
weather_loader  = WeatherDataLoader(cache_dir='data/external')
fuel_loader     = FuelPriceLoader(cache_dir='data/external')
neighbor_loader = NeighborGenerationLoader(cache_dir='data/external/neighbors')

# ── ENTSO-E (optional — requires free API key) ───────────────────────────────
_ENTSOE_API_KEY = os.environ.get('ENTSOE_API_KEY')
entsoe_loader   = None
if _ENTSOE_API_KEY:
    try:
        from src.data.entsoe_loader import EntsoELoader
        entsoe_loader = EntsoELoader(api_key=_ENTSOE_API_KEY, cache_dir='data/external/entsoe')
        print("\n✅ ENTSO-E API key found — cross-border flows, generation mix and forecasts will be included.")
    except Exception as exc:
        print(f"\n⚠️  ENTSO-E loader init failed: {exc}")
else:
    print("\n⚠️  ENTSOE_API_KEY not set — running without ENTSO-E data.")
    print("   Get a free key at https://transparency.entsoe.eu/ to add ~30 powerful features.\n")

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

    # ── Neighbor country generation (France nuclear, Portugal hydro) ──────────
    try:
        print(f"\n  Loading neighbor generation for {zone}...")
        neighbor_df = neighbor_loader.load_all(zone, START_DATE, END_DATE, use_cache=True)
        if not neighbor_df.empty:
            neighbor_df['timestamp'] = pd.to_datetime(neighbor_df['timestamp'], utc=True)
            nbr_cols = [c for c in neighbor_df.columns if c != 'timestamp']
            merged = merged.merge(neighbor_df[['timestamp'] + nbr_cols], on='timestamp', how='left')
            merged[nbr_cols] = merged[nbr_cols].ffill().bfill()
            print(f"      Neighbor columns added: {nbr_cols}")
    except Exception as exc:
        print(f"      ⚠️  Neighbor generation merge failed: {exc}")

    # ── ENTSO-E merge (optional, requires API key) ────────────────────────────
    if entsoe_loader is not None:
        try:
            entsoe_df = entsoe_loader.load_all_features(zone, START_DATE, END_DATE, use_cache=True)
            entsoe_df['timestamp'] = pd.to_datetime(entsoe_df['timestamp'], utc=True)
            entsoe_cols = [c for c in entsoe_df.columns if c != 'timestamp']
            merged = merged.merge(entsoe_df[['timestamp'] + entsoe_cols], on='timestamp', how='left')
            # Forward-fill cross-border flows (can have gaps around midnight)
            merged[entsoe_cols] = merged[entsoe_cols].ffill().bfill()
            print(f"      ENTSO-E features added: {entsoe_cols}")
        except Exception as exc:
            print(f"      ⚠️  ENTSO-E merge failed: {exc}")

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
    ts_col = pd.Series(df_feat_save['timestamp'].values, dtype='datetime64[ns, UTC]')
    df_feat_save['timestamp'] = ts_col.dt.tz_convert(TIMEZONE[zone]).apply(lambda x: x.isoformat())
    df_feat_save.to_csv(feat_path, index=False)
    print(f"      Saved engineered → {feat_path}")

    # Save training data (X + y combined)
    train_path = Path(f'data/engineered/{zone.lower()}_training_data.csv')
    df_feat_save.drop(columns=['timestamp']).to_csv(train_path, index=False)
    print(f"      Saved training   → {train_path}")

print("\n" + "="*65)
print("  DONE — now re-run notebook 04 to train with the full dataset")
print("="*65)
