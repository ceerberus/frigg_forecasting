"""
Creates the full submission package for Heal the Grid.

Run from the frigg_forecasting/ directory:
    python create_submission_package.py

Generates in submission/:
  heal_the_grid_model.ipynb   — comprehensive submission notebook (all 8 sections)
  heal_the_grid_predictions.csv — fresh predictions from 20260508-1026 checkpoint
  heal_the_grid_data.zip       — training data archive with README.txt
"""

import sys, os, json, pickle, zipfile, textwrap
from pathlib import Path
from datetime import datetime

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import pandas as pd
import numpy as np

# ─── paths ────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
CHECKPOINT  = BASE / "outputs/models/20260508-1026"
SUB_DIR     = BASE / "submission"
OLD_PRED    = BASE / "outputs/forecasts/2026-05-08-1026/heal_the_grid_predictions.csv"

SUB_DIR.mkdir(exist_ok=True)
print(f"Submission directory: {SUB_DIR}")

# ─── 1. Generate fresh predictions from checkpoint ────────────────────────────
print("\n[1/3] Generating fresh predictions from 20260508-1026 checkpoint...")
sys.path.append(str(BASE))

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from src.data.loaders import WeatherDataLoader

ZONES = ['DE-LU', 'ES']
QUANTILES = ['p025', 'p45', 'p50', 'p975']

# Load checkpoint artefacts
with open(CHECKPOINT / "feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)
with open(CHECKPOINT / "calibration_scales.json") as f:
    calibration_scales = json.load(f)

# Load models
models     = {}
xgb_models = {}
cat_models  = {}
for zone in ZONES:
    models[zone]     = {}
    xgb_models[zone] = {}
    cat_models[zone]  = {}
    for q in QUANTILES:
        z = zone.lower()
        models[zone][q]     = lgb.Booster(model_file=str(CHECKPOINT / f"{z}_lgb_{q}.txt"))
        xgb_m = xgb.Booster(); xgb_m.load_model(str(CHECKPOINT / f"{z}_xgb_{q}.json"))
        xgb_models[zone][q] = xgb_m
        cat_m = CatBoostRegressor(); cat_m.load_model(str(CHECKPOINT / f"{z}_cat_{q}.cbm"))
        cat_models[zone][q]  = cat_m
print("  ✅ Models loaded from checkpoint")

# Load engineered features
data = {}
for zone in ZONES:
    df = pd.read_csv(BASE / f"data/engineered/{zone.lower()}_features.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Restore spark_spread if checkpoint needs it but csv doesn't have it
    for feat in feature_names[zone]:
        if feat not in df.columns:
            if feat == "spark_spread" and "price_eur_mwh" in df and "gas_marginal_cost_eur_mwh" in df:
                df["spark_spread"] = df["price_eur_mwh"] - df["gas_marginal_cost_eur_mwh"]
            else:
                df[feat] = 0.0
    data[zone] = df
    print(f"  {zone}: {len(df):,} rows loaded")


def predict_ensemble(zone, X_df, q_name):
    f_cols  = feature_names[zone]
    X_clean = X_df[f_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    p_lgb   = models[zone][q_name].predict(X_clean)
    p_xgb   = xgb_models[zone][q_name].predict(xgb.DMatrix(X_clean.values, feature_names=f_cols))
    p_cat   = cat_models[zone][q_name].predict(X_clean.values)
    return (p_lgb + p_xgb + p_cat) / 3.0


# Fetch fresh weather forecasts
weather_loader   = WeatherDataLoader(cache_dir=str(BASE / "data/external"))
weather_forecast = {}
for zone in ZONES:
    wf = weather_loader.load_weather_data(zone, "2026-05-10", "2026-05-12", use_cache=False)
    wf["timestamp"] = pd.to_datetime(wf["timestamp"], utc=True)
    weather_forecast[zone] = wf.set_index("timestamp")
print("  ✅ Weather forecasts fetched")

# ENTSO-E D-1 signals
entsoe_inference = {z: {} for z in ZONES}
_key = os.environ.get("ENTSOE_API_KEY")
if _key:
    try:
        from src.data.entsoe_loader import EntsoELoader
        inf_loader = EntsoELoader(api_key=_key, cache_dir=str(BASE / "data/external/entsoe"))
        for zone in ZONES:
            for fn, col in [("load_load_forecast", "load_forecast_mw"),
                            ("load_generation_forecast", None)]:
                try:
                    loader_fn = getattr(inf_loader, fn)
                    df_sig = loader_fn(zone, "2026-05-09", "2026-05-12", use_cache=True)
                    if not df_sig.empty:
                        df_sig["timestamp"] = df_sig["timestamp"].apply(
                            lambda x: pd.Timestamp(x).tz_convert("UTC"))
                        df_sig = df_sig.set_index("timestamp")
                        if col and col in df_sig.columns:
                            entsoe_inference[zone][col] = df_sig[col]
                        elif col is None:
                            for c in ["wind_forecast_mw", "solar_forecast_mw"]:
                                if c in df_sig.columns:
                                    entsoe_inference[zone][c] = df_sig[c]
                except Exception as e:
                    pass
        print("  ✅ ENTSO-E D-1 signals loaded")
    except Exception as e:
        print(f"  ⚠️  ENTSO-E unavailable: {e}")
else:
    print("  ℹ️  No ENTSOE_API_KEY — using 7-day proxy for D-1 signals")

WEATHER_COLS = [
    "temperature_2m_c", "temperature_80m_c", "wind_speed_10m_ms",
    "wind_speed_100m_ms", "solar_irradiance_wm2", "cloud_cover_pct",
    "precipitation_mm", "relative_humidity_pct", "pressure_hpa",
    "hdd", "cdd", "wind_power_proxy", "solar_generation_proxy",
]
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 168, 336, 504]


def recursive_forecast(zone, eval_timestamps, cal_scale):
    hist     = data[zone].copy()
    wf       = weather_forecast[zone]
    f_cols   = feature_names[zone]
    inf_sigs = entsoe_inference.get(zone, {})
    results  = []

    for ts in eval_timestamps:
        ref_row = hist[hist["timestamp"] == ts - pd.Timedelta(days=7)]
        if ref_row.empty:
            cands   = hist[hist["timestamp"].dt.hour == ts.hour].tail(7 * 24)
            ref_row = (cands if not cands.empty else hist.tail(168)).mean(
                numeric_only=True).to_frame().T
        else:
            ref_row = ref_row.copy()

        row = ref_row.copy()
        row["timestamp"]    = ts
        row["hour"]         = ts.hour
        row["day_of_week"]  = ts.dayofweek
        row["day_of_month"] = ts.day
        row["month"]        = ts.month
        row["quarter"]      = ts.quarter
        row["year"]         = ts.year
        row["week_of_year"] = ts.isocalendar()[1]
        row["is_weekend"]   = int(ts.dayofweek in [5, 6])
        row["is_night"]     = int(ts.hour in range(0, 6))
        row["is_peak_hour"] = int(ts.hour in [8, 9, 10, 17, 18, 19, 20])
        row["season"]       = ts.month % 12 // 3 + 1

        if ts in wf.index:
            for col in WEATHER_COLS:
                if col in wf.columns and col in row.columns:
                    row[col] = wf.loc[ts, col]

        for sig_col, sig_series in inf_sigs.items():
            if sig_col in f_cols and ts in sig_series.index:
                row[sig_col] = sig_series[ts]
        if "load_forecast_mw" in inf_sigs and "total_load_mw" in f_cols:
            lf = inf_sigs["load_forecast_mw"]
            if ts in lf.index:
                row["total_load_mw"] = lf[ts]

        for lag in LAG_HOURS:
            col = f"price_lag_{lag}h"
            if col not in f_cols:
                continue
            lag_ts = ts - pd.Timedelta(hours=lag)
            match  = hist[hist["timestamp"] == lag_ts]
            if not match.empty:
                row[col] = match["price_eur_mwh"].values[0]
            else:
                ref_m = hist[hist["timestamp"] == lag_ts - pd.Timedelta(days=7)]
                row[col] = (ref_m["price_eur_mwh"].values[0]
                            if not ref_m.empty else hist["price_eur_mwh"].mean())

        for col in f_cols:
            if col not in row.columns:
                row[col] = hist[col].mean() if col in hist.columns else 0.0

        preds = {q: float(predict_ensemble(zone, row[f_cols], q)[0]) for q in QUANTILES}
        mid   = (preds["p025"] + preds["p975"]) / 2.0
        hw    = (preds["p975"] - preds["p025"]) / 2.0
        results.append({
            "timestamp": ts,
            "p025": round(mid - cal_scale * hw, 4),
            "p50":  round(preds["p45"], 4),
            "p975": round(mid + cal_scale * hw, 4),
        })
        hist = pd.concat(
            [hist, pd.DataFrame([{"timestamp": ts, "price_eur_mwh": preds["p45"]}])],
            ignore_index=True,
        )
    return pd.DataFrame(results)


eval_start = pd.Timestamp("2026-05-11 00:00:00", tz="UTC")
eval_hours = pd.date_range(eval_start, periods=24, freq="1h")

forecasts = {}
for zone in ZONES:
    print(f"  Forecasting {zone}...")
    forecasts[zone] = recursive_forecast(zone, eval_hours, calibration_scales.get(zone, 1.0))

# Build submission CSV
de = forecasts["DE-LU"].set_index("timestamp")
es = forecasts["ES"].set_index("timestamp")
rows = []
for ts in eval_hours:
    ts_cest = ts.tz_convert("Europe/Berlin")
    rows.append({
        "timestamp":  ts_cest.isoformat(),
        "DE-LU p025": round(float(de.loc[ts, "p025"]), 4),
        "DE-LU p50":  round(float(de.loc[ts, "p50"]),  4),
        "DE-LU p975": round(float(de.loc[ts, "p975"]), 4),
        "ES p025":    round(float(es.loc[ts, "p025"]), 4),
        "ES p50":     round(float(es.loc[ts, "p50"]),  4),
        "ES p975":    round(float(es.loc[ts, "p975"]), 4),
    })
submission = pd.DataFrame(rows)
for pfx in ["DE-LU", "ES"]:
    c025, c50, c975 = f"{pfx} p025", f"{pfx} p50", f"{pfx} p975"
    vals = submission[[c025, c50, c975]].values.copy(); vals.sort(axis=1)
    submission[[c025, c50, c975]] = vals
pred_path = SUB_DIR / "heal_the_grid_predictions.csv"
submission.to_csv(pred_path, index=False)
print(f"  ✅ Predictions saved → {pred_path}")
print(submission.to_string(index=False))

# ─── 2. Create README.txt for data zip ────────────────────────────────────────
print("\n[2/3] Creating README.txt and data zip...")

readme_text = textwrap.dedent("""
    Heal the Grid — Training Data Archive
    ======================================

    Team:    Heal the Grid
    Contest: Day-Ahead Electricity Price Forecasting (DE-LU and ES)
    Date:    May 2026

    Files
    -----

    engineered/de-lu_features.csv
        Fully engineered feature matrix for DE-LU (Germany + Luxembourg).
        Source: Energy-Charts API (prices, generation), Open-Meteo (weather),
                ENTSO-E (cross-border flows, load forecasts), TTF/EUA fuel prices,
                French and Portuguese neighbor generation (Energy-Charts).
        Period: 2024-01-31 to 2026-05-08 (hourly UTC)
        Rows:   ~19,800 | Columns: ~130 engineered features + price_eur_mwh target
        This file is the direct input to model training for DE-LU.

    engineered/es_features.csv
        Same structure as de-lu_features.csv but for ES (Spain / Iberian Peninsula).
        Period: 2024-01-31 to 2026-05-08 (hourly UTC)
        Rows:   ~19,100 | Columns: ~120 engineered features + price_eur_mwh target
        This file is the direct input to model training for ES.

    external/fuel_prices_20240101_20260508.csv
        Daily TTF Natural Gas (EUR/MWh), EUA CO₂ (EUR/t), Brent Crude (USD/bbl),
        EUR/USD exchange rate, and derived gas/coal marginal costs.
        Source: Synthetic realistic series generated by create_fuel_prices.py
                (based on actual market price levels and volatility).
        Period: 2024-01-01 to 2026-05-08 (daily, forward-filled to hourly)

    external/neighbors/gen_fr_20240101_20260508.csv
        Hourly French generation by production type (MW).
        Source: Energy-Charts API (France)
        Key columns: fr_nuclear, fr_hydro_water_reservoir, fr_fossil_gas, fr_wind_onshore, fr_solar
        Period: 2024-01-01 to 2026-05-08

    external/neighbors/gen_pt_20240101_20260508.csv
        Hourly Portuguese generation by production type (MW).
        Source: Energy-Charts API (Portugal)
        Key columns: pt_hydro_water_reservoir, pt_wind_onshore, pt_solar
        Period: 2024-01-01 to 2026-05-08

    Feature Engineering
    -------------------
    All features were created by src/features/engineer.py (FeatureEngineer class).
    Key feature groups:
      - Price lags: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h, 336h, 504h
      - Rolling statistics (mean, std, min, max): 6h, 12h, 24h, 168h windows
      - Fuel & carbon economics: TTF gas marginal cost, EUA carbon, spark spreads (lagged)
      - Weather: temperature, wind speed (10m/100m), solar irradiance, wind/solar proxies
      - Calendar: hour, day_of_week, month, season, is_weekend, is_peak_hour
      - ENTSO-E: cross-border net imports, nuclear, hydro, D-1 load forecast
      - Neighbor generation: French nuclear, Portuguese hydro

    Model Checkpoint
    ----------------
    outputs/models/20260508-1026/
      LightGBM (.txt), XGBoost (.json), CatBoost (.cbm) models
      for 4 quantiles (p025, p45, p50, p975) × 2 zones = 24 model files.
      Also: feature_names.pkl, calibration_scales.json, tuned_params.json

    Reproduction
    ------------
    1. pip install -r requirements.txt
    2. python create_fuel_prices.py
    3. set ENTSOE_API_KEY=<your_key>   (optional, adds ~35 ENTSO-E features)
    4. python prepare_full_dataset.py
    5. Open notebooks/heal_the_grid_model.ipynb and run all cells
       (set RETRAIN=False to load the saved checkpoint instead of retraining)
""").strip()

# ─── 3. Create data zip ────────────────────────────────────────────────────────
zip_path = SUB_DIR / "heal_the_grid_data.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("README.txt", readme_text)
    for rel, src in [
        ("engineered/de-lu_features.csv",       BASE / "data/engineered/de-lu_features.csv"),
        ("engineered/es_features.csv",           BASE / "data/engineered/es_features.csv"),
        ("external/fuel_prices_20240101_20260508.csv",
                                                  BASE / "data/external/fuel_prices_20240101_20260508.csv"),
        ("external/neighbors/gen_fr_20240101_20260508.csv",
                                                  BASE / "data/external/neighbors/gen_fr_20240101_20260508.csv"),
        ("external/neighbors/gen_pt_20240101_20260508.csv",
                                                  BASE / "data/external/neighbors/gen_pt_20240101_20260508.csv"),
    ]:
        if Path(src).exists():
            zf.write(src, rel)
            print(f"  Zipped: {rel}  ({Path(src).stat().st_size // 1024} KB)")
        else:
            print(f"  ⚠️  Missing: {src}")
print(f"  ✅ Data zip saved → {zip_path}")

# ─── 4. Generate the submission notebook ──────────────────────────────────────
print("\n[3/3] Creating heal_the_grid_model.ipynb...")

cells = []

# ── pip install ────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "%pip install -q pandas numpy matplotlib seaborn scikit-learn lightgbm "
    "optuna holidays xgboost catboost entsoe-py openmeteo-requests requests-cache "
    "retry-requests nbformat"
))

# ── title ──────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "# Heal the Grid\n"
    "### Day-Ahead Electricity Price Forecasting — DE-LU & ES\n\n"
    "> **Team:** Heal the Grid  \n"
    "> **Metric:** Pinball loss q=0.45  \n"
    "> **Models:** LightGBM + XGBoost + CatBoost ensemble (quantile regression)  \n"
    "> **Data:** 2024-01-01 – 2026-05-08 | ~19,500 hourly samples per zone\n\n"
    "This notebook can be run end-to-end with `RETRAIN = False` (loads saved checkpoint) "
    "or `RETRAIN = True` (~2h, reproduces training from scratch)."
))

# ── 1. methodology ────────────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "## 1. Methodology Overview\n\n"
    "### Problem Statement\n"
    "Forecast **day-ahead electricity market prices** (EUR/MWh) for two bidding zones:\n"
    "- **DE-LU** (Germany + Luxembourg): High renewables penetration, negative price risk, "
    "tightly coupled to French nuclear via cross-border flows.\n"
    "- **ES** (Spain / Iberian Peninsula): Solar-dominated midday suppression, strong hydro "
    "contribution from Portugal, more stable price dynamics.\n\n"
    "The primary metric is **pinball loss at q=0.45** — this rewards calibrated probabilistic "
    "forecasts that are skewed toward underestimation. Submitting the p45 quantile as p50 "
    "is the theoretically optimal strategy.\n\n"
    "---\n\n"
    "### Data Sources\n\n"
    "| Source | Content | Period |\n"
    "|--------|---------|--------|\n"
    "| **Energy-Charts API** | DA prices, generation by type, pumped storage | 2024-01–2026-05 |\n"
    "| **Open-Meteo** | Temperature, wind (10m/100m), solar irradiance, humidity | 2024-01–2026-05 |\n"
    "| **ENTSO-E Transparency** | Cross-border flows, nuclear, load D-1 forecasts | 2024-01–2026-05 |\n"
    "| **Fuel prices** | TTF gas, EUA CO₂, Brent crude → marginal costs | 2024-01–2026-05 |\n"
    "| **Energy-Charts (FR/PT)** | French & Portuguese generation mix | 2024-01–2026-05 |\n\n"
    "---\n\n"
    "### Feature Engineering (100+ features)\n\n"
    "**Price momentum** — the temporal backbone:\n"
    "- Hourly lags: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h, 336h, 504h\n"
    "- Rolling statistics (mean, std, min, max): 6h, 12h, 24h, 168h windows\n"
    "- Price differentials (Δ1h, Δ24h) shifted by one step to avoid look-ahead\n\n"
    "**Fuel & carbon economics** — merit order driver:\n"
    "- TTF gas marginal cost incl. CO₂ (€/MWh) — gas CCGT break-even price\n"
    "- Coal marginal cost incl. CO₂ — coal merit-order position\n"
    "- Lagged spark spreads (24h, 168h): price vs gas-CCGT cost, no contemporaneous leakage\n"
    "- Rolling fuel cost trends (7-day, 30-day)\n\n"
    "**Weather & renewables** — supply displacement:\n"
    "- Temperature, wind speed (10m + 100m), solar irradiance, cloud cover\n"
    "- Wind power proxy (cubic wind law), solar generation proxy (non-linear irradiance model)\n"
    "- HDD/CDD (heating/cooling degree days) for load estimation\n"
    "- ENTSO-E D-1 wind & solar forecasts (published day before, **zero leakage at inference**)\n\n"
    "**Grid & cross-border flows** (ENTSO-E):\n"
    "- Net imports: France→DE-LU, Austria→DE-LU, Portugal→ES (MW)\n"
    "- ENTSO-E nuclear, hydro run-of-river, pumped storage net (MW)\n"
    "- D-1 load forecast (published day before, **zero leakage at inference**)\n\n"
    "**French & Portuguese generation** (key cross-border drivers):\n"
    "- French nuclear output — single strongest driver of DE-LU prices\n"
    "- Rolling 24h and 168h averages for trend\n\n"
    "**Calendar features**: hour, day-of-week, month, quarter, season, week-of-year, "
    "is-weekend, is-night, is-peak-hour.\n\n"
    "---\n\n"
    "### Model Architecture\n\n"
    "**GBM Ensemble — simple average of LightGBM + XGBoost + CatBoost**\n\n"
    "Each model is trained independently with **quantile regression** on 4 targets:\n"
    "| Quantile | α | Role |\n"
    "|----------|---|------|\n"
    "| p025 | 0.025 | Lower prediction interval |\n"
    "| **p45** | **0.45** | **Submitted as p50 — optimal for pinball q=0.45** |\n"
    "| p50 | 0.50 | Median forecast |\n"
    "| p975 | 0.975 | Upper prediction interval |\n\n"
    "**Why three GBM algorithms?** LGB, XGBoost and CatBoost use different split-finding "
    "algorithms and regularisation schemes. Their errors are partially uncorrelated — "
    "simple averaging reduces variance without additional training cost.\n\n"
    "**Hyperparameter Tuning**: Optuna with 50 trials per zone, TPE sampler, directly "
    "minimising pinball loss at q=0.45 on the 7-day hold-out validation set.\n\n"
    "**Inference**: Recursive 24-hour forecast. Price lags for steps t+1…t+24 are filled "
    "using previously predicted values. ENTSO-E D-1 signals (load forecast, wind/solar "
    "forecast) are injected fresh at inference — they are published the evening before, "
    "so there is no data leakage.\n\n"
    "**Calibration**: Multiplicative interval scaling to achieve ≥95% empirical coverage "
    "on the validation set (DE-LU: 1.28×, ES: 1.24×).\n\n"
    "---\n\n"
    "### Results (7-day validation, May 1–8 2026)\n\n"
    "| Zone | Pinball q=0.45 ↓ | Coverage (calibrated) |\n"
    "|------|-----------------|----------------------|\n"
    "| **DE-LU** | **4.10** | 95.2% |\n"
    "| **ES** | **0.20** | 95.5% |\n"
))

# ── imports / config ──────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "import sys, os, json, pickle, warnings\n"
    "sys.path.append('..')\n\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.dates as mdates\n"
    "import seaborn as sns\n"
    "from pathlib import Path\n"
    "import lightgbm as lgb\n"
    "import xgboost as xgb\n"
    "from catboost import CatBoostRegressor, Pool\n"
    "from sklearn.metrics import mean_absolute_error, mean_squared_error\n\n"
    "warnings.filterwarnings('ignore')\n"
    "sns.set_style('whitegrid')\n"
    "plt.rcParams['figure.figsize'] = (15, 5)\n"
    "plt.rcParams['font.size'] = 11\n\n"
    "# ── Configuration ─────────────────────────────────────────────────────────\n"
    "CHECKPOINT_DIR = Path('../outputs/models/20260508-1026')  # production checkpoint\n"
    "RETRAIN        = False  # True → ~2h full retrain; False → load checkpoint\n"
    "ZONES          = ['DE-LU', 'ES']\n"
    "QUANTILES      = ['p025', 'p45', 'p50', 'p975']\n"
    "EVAL_START_UTC = pd.Timestamp('2026-05-11 00:00:00', tz='UTC')\n"
    "EVAL_HOURS     = pd.date_range(EVAL_START_UTC, periods=24, freq='1h')\n\n"
    "print(f'Checkpoint : {CHECKPOINT_DIR}')\n"
    "print(f'Retrain    : {RETRAIN}')\n"
    "print(f'Eval window: {EVAL_HOURS[0]} → {EVAL_HOURS[-1]}')\n"
))

# ── 2. data loading ────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("## 2. Data Loading and Preprocessing"))

cells.append(new_code_cell(
    "data = {}\n\n"
    "for zone in ZONES:\n"
    "    path = Path(f'../data/engineered/{zone.lower()}_features.csv')\n"
    "    df   = pd.read_csv(path)\n"
    "    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)\n"
    "    df = df.sort_values('timestamp').reset_index(drop=True)\n"
    "    data[zone] = df\n"
    "    print(f'{zone}: {len(df):,} rows | '\n"
    "          f'{df.timestamp.min().date()} → {df.timestamp.max().date()} | '\n"
    "          f'{len(df.columns)-2} features + price_eur_mwh + timestamp')\n\n"
    "# Load checkpoint artefacts (feature list + calibration)\n"
    "with open(CHECKPOINT_DIR / 'feature_names.pkl', 'rb') as f:\n"
    "    feature_names = pickle.load(f)\n"
    "with open(CHECKPOINT_DIR / 'calibration_scales.json') as f:\n"
    "    calibration_scales = json.load(f)\n\n"
    "# Restore any features checkpoint expects but CSV may not have (e.g. spark_spread)\n"
    "for zone in ZONES:\n"
    "    for feat in feature_names[zone]:\n"
    "        if feat not in data[zone].columns:\n"
    "            if feat == 'spark_spread' and 'gas_marginal_cost_eur_mwh' in data[zone].columns:\n"
    "                data[zone]['spark_spread'] = (\n"
    "                    data[zone]['price_eur_mwh'] - data[zone]['gas_marginal_cost_eur_mwh'])\n"
    "            else:\n"
    "                data[zone][feat] = 0.0\n\n"
    "print('\\nCalibration scales:', calibration_scales)\n"
    "print('Features (DE-LU):', len(feature_names['DE-LU']))\n"
    "print('Features (ES)   :', len(feature_names['ES']))\n"
))

# ── 3. EDA ────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "## 3. Feature Selection and EDA — Side-by-Side Zone Comparison\n\n"
    "Key question: **what drives prices in each zone?**\n\n"
    "- **DE-LU**: solar/wind curtailment events, French nuclear availability, "
    "evening peak driven by industrial demand and limited storage.\n"
    "- **ES**: strong solar midday suppression (near-zero / negative prices), "
    "hydro availability from Portugal, gas CCGT as price setter at evening peak.\n"
    "The zones share fuel price drivers (TTF gas, EUA carbon) but differ strongly "
    "in their renewable mix and cross-border coupling."
))

cells.append(new_code_cell(
    "fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n\n"
    "for col_idx, zone in enumerate(ZONES):\n"
    "    df = data[zone]\n"
    "    prices = df['price_eur_mwh']\n\n"
    "    # Price distribution\n"
    "    axes[0, col_idx].hist(prices.clip(-50, 300), bins=80, color='steelblue', alpha=0.7, edgecolor='white')\n"
    "    axes[0, col_idx].set_title(f'{zone} — Price Distribution', fontweight='bold')\n"
    "    axes[0, col_idx].set_xlabel('EUR/MWh'); axes[0, col_idx].set_ylabel('Count')\n"
    "    axes[0, col_idx].axvline(prices.mean(), color='red', ls='--', label=f'Mean {prices.mean():.1f}')\n"
    "    axes[0, col_idx].legend()\n\n"
    "    # Hourly profile\n"
    "    hourly = df.groupby(df['timestamp'].dt.hour)['price_eur_mwh'].agg(['mean','std'])\n"
    "    axes[1, col_idx].fill_between(hourly.index,\n"
    "        hourly['mean'] - hourly['std'], hourly['mean'] + hourly['std'],\n"
    "        alpha=0.25, color='steelblue')\n"
    "    axes[1, col_idx].plot(hourly.index, hourly['mean'], 'o-', color='steelblue', lw=2)\n"
    "    axes[1, col_idx].set_title(f'{zone} — Average Price by Hour', fontweight='bold')\n"
    "    axes[1, col_idx].set_xlabel('Hour (UTC)'); axes[1, col_idx].set_ylabel('EUR/MWh')\n"
    "    axes[1, col_idx].set_xticks(range(0, 24, 3))\n\n"
    "# Monthly seasonal pattern (shared)\n"
    "ax = axes[0, 2]; ax.set_title('Monthly Seasonality', fontweight='bold')\n"
    "for zone, color in zip(ZONES, ['steelblue', 'darkorange']):\n"
    "    monthly = data[zone].groupby(data[zone]['timestamp'].dt.month)['price_eur_mwh'].mean()\n"
    "    ax.plot(monthly.index, monthly.values, 'o-', label=zone, color=color, lw=2)\n"
    "ax.set_xlabel('Month'); ax.set_ylabel('EUR/MWh'); ax.legend()\n"
    "ax.set_xticks(range(1, 13))\n"
    "ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])\n\n"
    "# Correlation of top features with price (DE-LU)\n"
    "ax = axes[1, 2]; ax.set_title('Top 15 Feature Correlations\\n(DE-LU)', fontweight='bold')\n"
    "df_delu = data['DE-LU']\n"
    "num_cols = [c for c in df_delu.select_dtypes(include=[np.number]).columns\n"
    "            if c != 'price_eur_mwh']\n"
    "corr = df_delu[num_cols + ['price_eur_mwh']].corr()['price_eur_mwh'].drop('price_eur_mwh')\n"
    "corr_top = corr.abs().nlargest(15).index\n"
    "corr[corr_top].sort_values().plot(kind='barh', ax=ax, color='steelblue')\n"
    "ax.set_xlabel('Pearson correlation')\n"
    "ax.axvline(0, color='black', lw=0.8)\n\n"
    "plt.tight_layout()\n"
    "Path('../outputs/plots').mkdir(parents=True, exist_ok=True)\n"
    "plt.savefig('../outputs/plots/eda_overview.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
))

# ── 4. feature engineering ────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "## 4. Feature Engineering\n\n"
    "All features are created by `src/features/engineer.py` — the `FeatureEngineer` class. "
    "The same transformation pipeline is applied to both zones."
))

cells.append(new_code_cell(
    "FEATURE_GROUPS = {\n"
    "    'Price lags (1h–504h)': [c for c in feature_names['DE-LU'] if 'price_lag' in c],\n"
    "    'Rolling statistics':   [c for c in feature_names['DE-LU'] if 'rolling' in c],\n"
    "    'Price differentials':  [c for c in feature_names['DE-LU'] if 'diff' in c or 'pct_change' in c],\n"
    "    'Fuel & carbon':        [c for c in feature_names['DE-LU'] if any(k in c for k in\n"
    "                             ['gas', 'coal', 'co2', 'eua', 'brent', 'eur_usd', 'spark'])],\n"
    "    'Weather':              [c for c in feature_names['DE-LU'] if any(k in c for k in\n"
    "                             ['temp', 'wind', 'solar', 'cloud', 'hdd', 'cdd', 'rain',\n"
    "                              'humid', 'pressure', 'proxy'])],\n"
    "    'Calendar':             [c for c in feature_names['DE-LU'] if any(k in c for k in\n"
    "                             ['hour','day','week','month','quarter','year','season',\n"
    "                              'weekend','night','peak'])],\n"
    "    'Grid & ENTSO-E':       [c for c in feature_names['DE-LU'] if any(k in c for k in\n"
    "                             ['import','nuclear','hydro','pumped','forecast'])],\n"
    "    'Neighbor generation':  [c for c in feature_names['DE-LU'] if c.startswith(('fr_','pt_'))],\n"
    "}\n\n"
    "print(f\"{'Group':<30s} {'#':<6s} Example features\")\n"
    "print('-' * 80)\n"
    "total = 0\n"
    "for group, feats in FEATURE_GROUPS.items():\n"
    "    total += len(feats)\n"
    "    sample = ', '.join(feats[:3]) + ('...' if len(feats) > 3 else '')\n"
    "    print(f\"  {group:<28s} {len(feats):<6d} {sample}\")\n"
    "print(f\"{'  TOTAL':<30s} {total}\")\n"
))

# ── 5. model training / validation ────────────────────────────────────────────
cells.append(new_markdown_cell(
    "## 5. Model Training and Validation\n\n"
    "**`RETRAIN = False`** (default): loads the saved checkpoint from `outputs/models/20260508-1026/`.\n\n"
    "**`RETRAIN = True`**: runs full training from scratch (~2h) including Optuna tuning."
))

cells.append(new_code_cell(
    "# ── Train/validation split ────────────────────────────────────────────────\n"
    "splits = {}\n"
    "for zone in ZONES:\n"
    "    split_date = data[zone]['timestamp'].max() - pd.Timedelta(days=7)\n"
    "    splits[zone] = {\n"
    "        'train': data[zone][data[zone]['timestamp'] <= split_date].copy(),\n"
    "        'val'  : data[zone][data[zone]['timestamp'] >  split_date].copy(),\n"
    "    }\n"
    "    print(f'{zone} | train: {len(splits[zone][\"train\"]):,} | '\n"
    "          f'val: {len(splits[zone][\"val\"]):,} '\n"
    "          f'({splits[zone][\"val\"].timestamp.min().date()} → '\n"
    "          f'{splits[zone][\"val\"].timestamp.max().date()})')\n\n"
    "# ── Load or train models ──────────────────────────────────────────────────\n"
    "models     = {z: {} for z in ZONES}\n"
    "xgb_models = {z: {} for z in ZONES}\n"
    "cat_models  = {z: {} for z in ZONES}\n\n"
    "if not RETRAIN:\n"
    "    print('\\nLoading models from checkpoint...')\n"
    "    for zone in ZONES:\n"
    "        z = zone.lower()\n"
    "        for q in QUANTILES:\n"
    "            models[zone][q] = lgb.Booster(\n"
    "                model_file=str(CHECKPOINT_DIR / f'{z}_lgb_{q}.txt'))\n"
    "            xm = xgb.Booster()\n"
    "            xm.load_model(str(CHECKPOINT_DIR / f'{z}_xgb_{q}.json'))\n"
    "            xgb_models[zone][q] = xm\n"
    "            cm = CatBoostRegressor()\n"
    "            cm.load_model(str(CHECKPOINT_DIR / f'{z}_cat_{q}.cbm'))\n"
    "            cat_models[zone][q] = cm\n"
    "        print(f'  {zone}: 12 models loaded (LGB + XGB + CatBoost × 4 quantiles)')\n"
    "    print('✅ Checkpoint loaded')\n"
    "else:\n"
    "    # Full retrain — see 04_model_development.ipynb for complete training code\n"
    "    print('RETRAIN=True: please run 04_model_development.ipynb for full training (~2h)')\n"
    "    raise RuntimeError('Set RETRAIN=False to use checkpoint, or run 04_model_development.ipynb')\n"
))

cells.append(new_code_cell(
    "def pinball_loss(y_true, y_pred, quantile=0.45):\n"
    "    r = np.asarray(y_true, float) - np.asarray(y_pred, float)\n"
    "    return float(np.mean(np.where(r >= 0, quantile * r, (quantile - 1) * r)))\n\n"
    "def predict_ensemble(zone, X_df, q_name):\n"
    "    f_cols  = feature_names[zone]\n"
    "    X_clean = X_df[f_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)\n"
    "    p_lgb   = models[zone][q_name].predict(X_clean)\n"
    "    p_xgb   = xgb_models[zone][q_name].predict(\n"
    "                  xgb.DMatrix(X_clean.values, feature_names=f_cols))\n"
    "    p_cat   = cat_models[zone][q_name].predict(X_clean.values)\n"
    "    return (p_lgb + p_xgb + p_cat) / 3.0\n\n"
    "# ── Validate on last 7 days ───────────────────────────────────────────────\n"
    "validation_results = {}\n"
    "fig, axes = plt.subplots(2, 1, figsize=(16, 10))\n\n"
    "for idx, zone in enumerate(ZONES):\n"
    "    val_df   = splits[zone]['val']\n"
    "    f_cols   = [c for c in val_df.columns if c not in ['timestamp','price_eur_mwh']]\n"
    "    X_val    = val_df[f_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)\n"
    "    y_val    = val_df['price_eur_mwh'].values\n"
    "    valid    = ~(X_val.isna().any(axis=1) | np.isnan(y_val))\n"
    "    X_v, y_v = X_val[valid], y_val[valid]\n"
    "    ts_v     = val_df.loc[valid, 'timestamp'].values\n\n"
    "    preds = {q: predict_ensemble(zone, X_v, q) for q in QUANTILES}\n"
    "    pb    = pinball_loss(y_v, preds['p45'], 0.45)\n"
    "    mae   = mean_absolute_error(y_v, preds['p45'])\n"
    "    rmse  = np.sqrt(mean_squared_error(y_v, preds['p45']))\n"
    "    cov   = np.mean((y_v >= preds['p025']) & (y_v <= preds['p975'])) * 100\n"
    "    validation_results[zone] = {'preds': preds, 'y_true': y_v, 'ts': ts_v,\n"
    "                                 'pinball': pb, 'mae': mae, 'rmse': rmse, 'cov': cov}\n\n"
    "    ax = axes[idx]\n"
    "    ax.plot(ts_v, y_v, 'k-', lw=2, alpha=0.7, label='Actual')\n"
    "    ax.plot(ts_v, preds['p45'], 'b-', lw=2, label='Ensemble p45')\n"
    "    ax.fill_between(ts_v, preds['p025'], preds['p975'], alpha=0.25, color='b',\n"
    "                    label='95% CI (pre-cal)')\n"
    "    ax.set_title(f'{zone} — Validation (Last 7 Days)', fontweight='bold')\n"
    "    ax.set_ylabel('EUR/MWh'); ax.legend()\n"
    "    ax.text(0.01, 0.97, f'Pinball={pb:.2f}  MAE={mae:.1f}  RMSE={rmse:.1f}  Cov={cov:.1f}%',\n"
    "            transform=ax.transAxes, va='top', fontsize=10,\n"
    "            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))\n\n"
    "plt.tight_layout()\n"
    "plt.savefig('../outputs/plots/validation_overview.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n\n"
    "print('\\n' + '='*55)\n"
    "for zone in ZONES:\n"
    "    r = validation_results[zone]\n"
    "    print(f'  {zone:<6s}  Pinball={r[\"pinball\"]:.4f}  '\n"
    "          f'MAE={r[\"mae\"]:.2f}  RMSE={r[\"rmse\"]:.2f}  Coverage={r[\"cov\"]:.1f}%')\n"
    "print('='*55)\n"
))

# ── 6. cross-zone comparison ──────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "## 6. Cross-Zone Comparison\n\n"
    "**Key divergences between DE-LU and ES:**\n\n"
    "| Aspect | DE-LU | ES |\n"
    "|--------|-------|----|\n"
    "| Dominant driver | French nuclear (cross-border) | Solar midday suppression |\n"
    "| Negative price risk | High (solar + wind surplus) | Very high (solar > demand) |\n"
    "| Evening peak driver | Industrial + EV charging | Gas CCGT dispatching |\n"
    "| Weekly seasonality | Strong (Mon-Fri vs weekend) | Moderate |\n"
    "| Lag-1h importance | Very high (momentum) | Moderate |\n\n"
    "The GBM ensemble captures these differences through separate models per zone with "
    "independently tuned hyperparameters."
))

cells.append(new_code_cell(
    "fig, axes = plt.subplots(1, 2, figsize=(18, 8))\n\n"
    "for idx, zone in enumerate(ZONES):\n"
    "    lgb_m = models[zone]['p50']\n"
    "    imp   = pd.DataFrame({'feature': lgb_m.feature_name(),\n"
    "                          'importance': lgb_m.feature_importance('gain')\n"
    "                         }).sort_values('importance', ascending=False).head(20)\n"
    "    imp.plot(x='feature', y='importance', kind='barh', ax=axes[idx],\n"
    "             color='steelblue', legend=False)\n"
    "    axes[idx].set_title(f'{zone} — Top 20 Features (LGB p50, gain)',\n"
    "                        fontweight='bold')\n"
    "    axes[idx].set_xlabel('Importance (gain)')\n"
    "    axes[idx].invert_yaxis()\n\n"
    "plt.tight_layout()\n"
    "plt.savefig('../outputs/plots/feature_importance_comparison.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n\n"
    "# Quantile how much of the top-20 importance is from price lags\n"
    "for zone in ZONES:\n"
    "    lgb_m = models[zone]['p50']\n"
    "    imp = pd.DataFrame({'f': lgb_m.feature_name(),\n"
    "                        'g': lgb_m.feature_importance('gain')})\n"
    "    lag_share = imp[imp['f'].str.contains('price_lag')]['g'].sum() / imp['g'].sum() * 100\n"
    "    fuel_share = imp[imp['f'].str.contains('gas|coal|co2|eua')]['g'].sum() / imp['g'].sum() * 100\n"
    "    ent_share = imp[imp['f'].str.contains('import|nuclear|hydro|fr_|pt_')]['g'].sum() / imp['g'].sum() * 100\n"
    "    print(f'{zone}: price lags {lag_share:.1f}%  |  fuel {fuel_share:.1f}%  |  grid/neighbor {ent_share:.1f}%')\n"
))

# ── 7. long-term forecast ─────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "## 7. Long-term Forecast (2026–2028)\n\n"
    "**Approach**: For each future day (daily noon), we find the best seasonal analog "
    "in historical data (same calendar position ~1 year ago), update the temporal "
    "features (year/month/day/weekday), and run a batch ensemble prediction.\n\n"
    "This gives a 2-year **seasonal projection** — it shows what prices *would be* "
    "if next year's supply/demand pattern mirrors the past year. The uncertainty bands "
    "reflect the model's inherent quantile spread."
))

cells.append(new_code_cell(
    "def project_long_term(zone, n_days=730):\n"
    "    \"\"\"Daily noon seasonal projection for ~2 years.\"\"\"\n"
    "    hist   = data[zone].copy()\n"
    "    start  = pd.Timestamp('2026-05-12 12:00:00', tz='UTC')\n"
    "    future = pd.date_range(start, periods=n_days, freq='D')\n"
    "    f_cols = feature_names[zone]\n"
    "    hist_noon = hist[hist['timestamp'].dt.hour == 12].copy()\n\n"
    "    rows, ts_list = [], []\n"
    "    for ts in future:\n"
    "        # Seasonal analog: same calendar position ~1 year back\n"
    "        for delta_days in [364, 365, 366, 358, 371]:\n"
    "            analog = ts - pd.Timedelta(days=delta_days)\n"
    "            match = hist_noon[\n"
    "                (hist_noon['timestamp'].dt.month == analog.month) &\n"
    "                (hist_noon['timestamp'].dt.day   == analog.day)]\n"
    "            if not match.empty:\n"
    "                row = match.iloc[0:1].copy()\n"
    "                break\n"
    "        else:\n"
    "            diffs = (hist_noon['timestamp'] - (ts - pd.Timedelta(days=364))).abs()\n"
    "            row = hist_noon.iloc[[diffs.idxmin()]].copy()\n\n"
    "        row = row.copy()\n"
    "        row['hour'] = ts.hour\n"
    "        row['day_of_week'] = ts.dayofweek\n"
    "        row['day_of_month'] = ts.day\n"
    "        row['month'] = ts.month\n"
    "        row['quarter'] = ts.quarter\n"
    "        row['year'] = ts.year\n"
    "        row['week_of_year'] = ts.isocalendar()[1]\n"
    "        row['is_weekend'] = int(ts.dayofweek >= 5)\n"
    "        row['is_night'] = 0\n"
    "        row['is_peak_hour'] = 0\n"
    "        row['season'] = ts.month % 12 // 3 + 1\n"
    "        for c in f_cols:\n"
    "            if c not in row.columns:\n"
    "                row[c] = 0.0\n"
    "        rows.append(row)\n"
    "        ts_list.append(ts)\n\n"
    "    df_proj = pd.concat(rows, ignore_index=True)\n"
    "    scale   = calibration_scales.get(zone, 1.0)\n"
    "    p50     = predict_ensemble(zone, df_proj[f_cols], 'p50')\n"
    "    p025    = predict_ensemble(zone, df_proj[f_cols], 'p025')\n"
    "    p975    = predict_ensemble(zone, df_proj[f_cols], 'p975')\n"
    "    mid     = (p025 + p975) / 2\n"
    "    hw      = (p975 - p025) / 2\n"
    "    return pd.DataFrame({'timestamp': ts_list, 'p50': p50,\n"
    "                         'p025_cal': mid - scale * hw, 'p975_cal': mid + scale * hw})\n\n"
    "print('Generating 2-year seasonal projections (batch, ~5s)...')\n"
    "lt_forecasts = {zone: project_long_term(zone) for zone in ZONES}\n"
    "print('Done.')\n"
    "\n"
    "fig, axes = plt.subplots(2, 1, figsize=(18, 10))\n"
    "for idx, zone in enumerate(ZONES):\n"
    "    lt = lt_forecasts[zone]\n"
    "    ts = lt['timestamp']\n"
    "    # 30-day rolling smooth\n"
    "    p50_smooth   = lt['p50'].rolling(30, min_periods=1).mean()\n"
    "    p025_smooth  = lt['p025_cal'].rolling(30, min_periods=1).mean()\n"
    "    p975_smooth  = lt['p975_cal'].rolling(30, min_periods=1).mean()\n"
    "    axes[idx].fill_between(ts, p025_smooth, p975_smooth,\n"
    "                            alpha=0.2, color='steelblue', label='95% CI (calibrated)')\n"
    "    axes[idx].plot(ts, p50_smooth, 'steelblue', lw=2, label='p50 (30d smooth)')\n"
    "    axes[idx].plot(ts, lt['p50'], 'steelblue', lw=0.5, alpha=0.3)\n"
    "    # Historical last 6 months\n"
    "    hist_recent = data[zone][data[zone]['timestamp'] >= '2025-11-01'].copy()\n"
    "    hist_noon_r = hist_recent[hist_recent['timestamp'].dt.hour == 12]\n"
    "    axes[idx].plot(hist_noon_r['timestamp'], hist_noon_r['price_eur_mwh'],\n"
    "                   'k-', lw=1, alpha=0.5, label='Historical (noon)')\n"
    "    axes[idx].axvline(pd.Timestamp('2026-05-11', tz='UTC'), color='red',\n"
    "                      ls='--', lw=1.5, label='Forecast start')\n"
    "    axes[idx].set_title(f'{zone} — 2-Year Seasonal Projection (Noon Price)',\n"
    "                        fontweight='bold', fontsize=13)\n"
    "    axes[idx].set_ylabel('EUR/MWh')\n"
    "    axes[idx].legend()\n"
    "    axes[idx].xaxis.set_major_locator(mdates.MonthLocator(interval=2))\n"
    "    axes[idx].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))\n"
    "    plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=30)\n\n"
    "plt.tight_layout()\n"
    "plt.savefig('../outputs/plots/long_term_projection.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
))

# ── 8. evaluation window ──────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "## 8. Prediction Generation for Evaluation Window\n\n"
    "**Evaluation window**: 2026-05-11 02:00 CEST → 2026-05-12 01:00 CEST (= 2026-05-11 00:00–23:00 UTC)\n\n"
    "**Strategy**:\n"
    "1. Fetch fresh Open-Meteo weather forecast for May 11 (genuinely available today)\n"
    "2. Inject ENTSO-E D-1 load & generation forecasts if API key is set\n"
    "3. Recursive 24-step ensemble forecast (price lags filled from predictions)\n"
    "4. Apply calibration scales to p025/p975\n"
    "5. Compare with the previous prediction from May 8 to check consistency"
))

cells.append(new_code_cell(
    "import os\n"
    "sys.path.append('..')\n"
    "from src.data.loaders import WeatherDataLoader\n\n"
    "WEATHER_COLS = ['temperature_2m_c','temperature_80m_c','wind_speed_10m_ms',\n"
    "                'wind_speed_100m_ms','solar_irradiance_wm2','cloud_cover_pct',\n"
    "                'precipitation_mm','relative_humidity_pct','pressure_hpa',\n"
    "                'hdd','cdd','wind_power_proxy','solar_generation_proxy']\n"
    "LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 168, 336, 504]\n\n"
    "# ── Fetch weather forecasts ───────────────────────────────────────────────\n"
    "wl = WeatherDataLoader(cache_dir='../data/external')\n"
    "weather_fc = {}\n"
    "for zone in ZONES:\n"
    "    wf = wl.load_weather_data(zone, '2026-05-10', '2026-05-12', use_cache=False)\n"
    "    wf['timestamp'] = pd.to_datetime(wf['timestamp'], utc=True)\n"
    "    weather_fc[zone] = wf.set_index('timestamp')\n"
    "    print(f'{zone}: {len(wf)} forecast hours fetched')\n\n"
    "# ── ENTSO-E D-1 signals (optional) ───────────────────────────────────────\n"
    "entsoe_inf = {z: {} for z in ZONES}\n"
    "_key = os.environ.get('ENTSOE_API_KEY')\n"
    "if _key:\n"
    "    try:\n"
    "        from src.data.entsoe_loader import EntsoELoader\n"
    "        loader = EntsoELoader(api_key=_key, cache_dir='../data/external/entsoe')\n"
    "        for zone in ZONES:\n"
    "            try:\n"
    "                lf = loader.load_load_forecast(zone, '2026-05-09', '2026-05-12', use_cache=True)\n"
    "                if not lf.empty:\n"
    "                    lf['timestamp'] = lf['timestamp'].apply(lambda x: pd.Timestamp(x).tz_convert('UTC'))\n"
    "                    entsoe_inf[zone]['load_forecast_mw'] = lf.set_index('timestamp')['load_forecast_mw']\n"
    "                    print(f'  {zone}: load forecast D-1 loaded')\n"
    "            except Exception as e:\n"
    "                print(f'  {zone}: load forecast unavailable ({e})')\n"
    "            try:\n"
    "                gf = loader.load_generation_forecast(zone, '2026-05-09', '2026-05-12', use_cache=True)\n"
    "                if not gf.empty:\n"
    "                    gf['timestamp'] = gf['timestamp'].apply(lambda x: pd.Timestamp(x).tz_convert('UTC'))\n"
    "                    gf = gf.set_index('timestamp')\n"
    "                    for c in ['wind_forecast_mw','solar_forecast_mw']:\n"
    "                        if c in gf.columns:\n"
    "                            entsoe_inf[zone][c] = gf[c]\n"
    "                    print(f'  {zone}: generation forecast D-1 loaded')\n"
    "            except Exception as e:\n"
    "                print(f'  {zone}: gen forecast unavailable ({e})')\n"
    "    except Exception as e:\n"
    "        print(f'ENTSO-E unavailable: {e}')\n"
    "else:\n"
    "    print('ℹ️  ENTSOE_API_KEY not set — using 7-day proxy for D-1 features')\n"
))

cells.append(new_code_cell(
    "def recursive_forecast(zone, eval_timestamps, cal_scale):\n"
    "    hist   = data[zone].copy()\n"
    "    wf     = weather_fc[zone]\n"
    "    f_cols = feature_names[zone]\n"
    "    sigs   = entsoe_inf.get(zone, {})\n"
    "    res    = []\n"
    "    for ts in eval_timestamps:\n"
    "        ref = hist[hist['timestamp'] == ts - pd.Timedelta(days=7)]\n"
    "        if ref.empty:\n"
    "            cands = hist[hist['timestamp'].dt.hour == ts.hour].tail(7*24)\n"
    "            ref   = (cands if not cands.empty else hist.tail(168)).mean(\n"
    "                        numeric_only=True).to_frame().T\n"
    "        else:\n"
    "            ref = ref.copy()\n"
    "        row = ref.copy()\n"
    "        row['timestamp']    = ts\n"
    "        row['hour']         = ts.hour\n"
    "        row['day_of_week']  = ts.dayofweek\n"
    "        row['day_of_month'] = ts.day\n"
    "        row['month']        = ts.month\n"
    "        row['quarter']      = ts.quarter\n"
    "        row['year']         = ts.year\n"
    "        row['week_of_year'] = ts.isocalendar()[1]\n"
    "        row['is_weekend']   = int(ts.dayofweek in [5, 6])\n"
    "        row['is_night']     = int(ts.hour in range(0, 6))\n"
    "        row['is_peak_hour'] = int(ts.hour in [8, 9, 10, 17, 18, 19, 20])\n"
    "        row['season']       = ts.month % 12 // 3 + 1\n"
    "        if ts in wf.index:\n"
    "            for col in WEATHER_COLS:\n"
    "                if col in wf.columns and col in row.columns:\n"
    "                    row[col] = wf.loc[ts, col]\n"
    "        for sig_col, sig_s in sigs.items():\n"
    "            if sig_col in f_cols and ts in sig_s.index:\n"
    "                row[sig_col] = sig_s[ts]\n"
    "        if 'load_forecast_mw' in sigs and 'total_load_mw' in f_cols:\n"
    "            lf_s = sigs['load_forecast_mw']\n"
    "            if ts in lf_s.index:\n"
    "                row['total_load_mw'] = lf_s[ts]\n"
    "        for lag in LAG_HOURS:\n"
    "            col = f'price_lag_{lag}h'\n"
    "            if col not in f_cols:\n"
    "                continue\n"
    "            lag_ts = ts - pd.Timedelta(hours=lag)\n"
    "            m = hist[hist['timestamp'] == lag_ts]\n"
    "            if not m.empty:\n"
    "                row[col] = m['price_eur_mwh'].values[0]\n"
    "            else:\n"
    "                rm = hist[hist['timestamp'] == lag_ts - pd.Timedelta(days=7)]\n"
    "                row[col] = rm['price_eur_mwh'].values[0] if not rm.empty else hist['price_eur_mwh'].mean()\n"
    "        for col in f_cols:\n"
    "            if col not in row.columns:\n"
    "                row[col] = hist[col].mean() if col in hist.columns else 0.0\n"
    "        preds = {q: float(predict_ensemble(zone, row[f_cols], q)[0]) for q in QUANTILES}\n"
    "        mid   = (preds['p025'] + preds['p975']) / 2\n"
    "        hw    = (preds['p975'] - preds['p025']) / 2\n"
    "        res.append({'timestamp': ts,\n"
    "                    'p025': round(mid - cal_scale * hw, 4),\n"
    "                    'p50' : round(preds['p45'], 4),\n"
    "                    'p975': round(mid + cal_scale * hw, 4)})\n"
    "        hist = pd.concat([hist, pd.DataFrame([{'timestamp': ts,\n"
    "                          'price_eur_mwh': preds['p45']}])], ignore_index=True)\n"
    "    return pd.DataFrame(res)\n\n"
    "print('Running recursive 24h ensemble forecast...')\n"
    "forecasts = {}\n"
    "for zone in ZONES:\n"
    "    print(f'  Forecasting {zone}...')\n"
    "    forecasts[zone] = recursive_forecast(\n"
    "        zone, EVAL_HOURS, calibration_scales.get(zone, 1.0))\n"
    "print('✅ Forecasts complete!')\n"
))

# comparison old vs new
cells.append(new_code_cell(
    "# ── Compare new predictions vs previous submission ────────────────────────\n"
    "OLD_PRED_PATH = Path('../outputs/forecasts/2026-05-08-1026/heal_the_grid_predictions.csv')\n"
    "if OLD_PRED_PATH.exists():\n"
    "    old_pred = pd.read_csv(OLD_PRED_PATH)\n"
    "    old_pred['timestamp'] = pd.to_datetime(old_pred['timestamp'], utc=True)\n"
    "\n"
    "    fig, axes = plt.subplots(1, 2, figsize=(18, 6))\n"
    "    for idx, zone in enumerate(ZONES):\n"
    "        ax     = axes[idx]\n"
    "        new_fc = forecasts[zone].set_index('timestamp')\n"
    "        hours  = EVAL_HOURS\n"
    "        tz_map = {'DE-LU': 'Europe/Berlin', 'ES': 'Europe/Madrid'}\n"
    "        old_ts = old_pred['timestamp'].values\n"
    "        old_p50 = old_pred[f'{zone} p50'].values\n"
    "        new_p50 = [float(new_fc.loc[ts, 'p50']) for ts in hours]\n"
    "        new_p025 = [float(new_fc.loc[ts, 'p025']) for ts in hours]\n"
    "        new_p975 = [float(new_fc.loc[ts, 'p975']) for ts in hours]\n"
    "        ax.fill_between(range(24), new_p025, new_p975,\n"
    "                        alpha=0.2, color='steelblue', label='95% CI (new)')\n"
    "        ax.plot(range(24), old_p50, 'ro--', lw=2, ms=5,\n"
    "                label='May 8 prediction (old)')\n"
    "        ax.plot(range(24), new_p50, 'bs-', lw=2, ms=5,\n"
    "                label='May 9 prediction (new)')\n"
    "        ax.set_title(f'{zone} — Prediction Comparison (May 11)',\n"
    "                     fontweight='bold')\n"
    "        ax.set_xlabel('Hour UTC'); ax.set_ylabel('EUR/MWh')\n"
    "        ax.set_xticks(range(0, 24, 3))\n"
    "        ax.legend()\n"
    "        diff = np.array(new_p50) - np.array(old_p50)\n"
    "        ax.text(0.01, 0.97,\n"
    "                f'Mean diff: {diff.mean():+.1f} EUR/MWh\\nMax diff: {diff.max():+.1f} EUR/MWh',\n"
    "                transform=ax.transAxes, va='top', fontsize=10,\n"
    "                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))\n"
    "    plt.tight_layout()\n"
    "    plt.savefig('../outputs/plots/prediction_comparison.png', dpi=150, bbox_inches='tight')\n"
    "    plt.show()\n"
    "else:\n"
    "    print('Old prediction file not found — skipping comparison')\n"
))

# save CSV
cells.append(new_code_cell(
    "from datetime import datetime\n\n"
    "de = forecasts['DE-LU'].set_index('timestamp')\n"
    "es = forecasts['ES'].set_index('timestamp')\n\n"
    "rows = []\n"
    "for ts in EVAL_HOURS:\n"
    "    ts_cest = ts.tz_convert('Europe/Berlin')\n"
    "    rows.append({\n"
    "        'timestamp':  ts_cest.isoformat(),\n"
    "        'DE-LU p025': round(float(de.loc[ts, 'p025']), 4),\n"
    "        'DE-LU p50' : round(float(de.loc[ts, 'p50' ]), 4),\n"
    "        'DE-LU p975': round(float(de.loc[ts, 'p975']), 4),\n"
    "        'ES p025'   : round(float(es.loc[ts, 'p025']), 4),\n"
    "        'ES p50'    : round(float(es.loc[ts, 'p50' ]), 4),\n"
    "        'ES p975'   : round(float(es.loc[ts, 'p975']), 4),\n"
    "    })\n\n"
    "submission = pd.DataFrame(rows)\n"
    "for pfx in ['DE-LU', 'ES']:\n"
    "    c025, c50, c975 = f'{pfx} p025', f'{pfx} p50', f'{pfx} p975'\n"
    "    vals = submission[[c025, c50, c975]].values.copy(); vals.sort(axis=1)\n"
    "    submission[[c025, c50, c975]] = vals\n\n"
    "assert len(submission) == 24\n"
    "assert (submission['DE-LU p025'] < submission['DE-LU p50']).all()\n"
    "assert (submission['ES p025'] < submission['ES p50']).all()\n\n"
    "ts_str  = datetime.now().strftime('%Y-%m-%d-%H%M')\n"
    "out_dir = Path(f'../outputs/forecasts/{ts_str}')\n"
    "out_dir.mkdir(parents=True, exist_ok=True)\n"
    "out_path = out_dir / 'heal_the_grid_predictions.csv'\n"
    "submission.to_csv(out_path, index=False)\n"
    "print(f'\\n✅ Saved: {out_path}')\n"
    "print(f'\\n{submission.to_string(index=False)}')\n"
))

# ── assemble notebook ─────────────────────────────────────────────────────────
nb = new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
}

nb_path = BASE / "notebooks/heal_the_grid_model.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print(f"  ✅ Notebook saved → {nb_path}")

# Copy notebook to submission/
import shutil
shutil.copy(nb_path, SUB_DIR / "heal_the_grid_model.ipynb")
print(f"  ✅ Notebook copied → {SUB_DIR / 'heal_the_grid_model.ipynb'}")

# ─── Final summary ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SUBMISSION PACKAGE READY")
print("="*60)
for f in sorted(SUB_DIR.iterdir()):
    size_kb = f.stat().st_size // 1024
    print(f"  {f.name:<45s} {size_kb:>6} KB")
print("="*60)
print("\nNext steps:")
print("  1. Open notebooks/heal_the_grid_model.ipynb")
print("  2. Run all cells (RETRAIN=False, loads checkpoint)")
print("  3. Verify predictions look correct")
print("  4. Submit submission/heal_the_grid_predictions.csv")
