# Electricity Price Forecasting Challenge

## Project Overview
Multi-horizon ensemble forecasting system for European electricity markets (DE-LU and ES zones).

## Day-ahead model (leakage-free)

`python scripts/run_day_ahead.py` downloads the data, tunes, backtests and replays the live
evaluation day. Results go to `outputs/backtest/metrics.json`.

**Why it exists.** The hackathon pipeline (notebooks 03/04) had target leakage: `price_diff_1h`,
`price_pct_change_*` and the rolling price windows included the price being predicted
(`price_lag_1h + price_diff_1h` reproduced the target exactly), and short price lags and realised
generation are not known at auction time. Validation scores were therefore meaningless, and at
forecast time those features had to be filled with last week's values.

**Method**
- *Information set* (`src/features/day_ahead.py`): for delivery day D, prices only up to day D-1,
  TSO day-ahead forecasts of load/solar/wind for D (Energy-Charts), calendar. One feature function
  for training and inference; `tests/test_day_ahead_features.py` asserts that perturbing prices
  from day D onwards leaves the features of day D unchanged.
- *Model* (`src/models/quantile_lgbm.py`): LightGBM quantile regression per zone for
  q = 0.025 / 0.45 / 0.975. The 0.45-quantile is the point forecast because it minimises expected
  pinball loss at q = 0.45 (the challenge metric). Hyperparameters tuned with Optuna (TPE) on a
  validation period (Nov 2024 – Feb 2025) that precedes the backtest.
- *Evaluation*: expanding-window rolling-origin backtest, 365 delivery days (2025-05-11 – 2026-05-10),
  every day forecast out-of-sample (refit every 14 days). Diebold-Mariano tests (HAC variance on
  daily losses) against naive benchmarks (same hour D-1, same hour D-7).
- *Intervals*: conformalized quantile regression (CQR), recalibrated daily on the previous 90 days
  of out-of-sample conformity scores.

**Backtest results (365 days, 8,758 hours per zone)**

| | DE-LU | ES |
|---|---|---|
| Pinball loss q=0.45, model | 5.61 | 4.69 |
| Pinball loss q=0.45, best naive (D-1) | 13.22 | 9.12 |
| Improvement vs. best naive | 58% | 49% |
| Diebold-Mariano p-value vs. best naive | < 1e-30 | < 1e-30 |
| rMAE vs. naive D-7 | 0.35 | 0.35 |
| 95% interval coverage, raw quantiles → CQR | 80.1% → 94.7% | 85.9% → 95.4% |
| Interval (Winkler) score, raw → CQR | 113.6 → 93.2 | 72.2 → 65.2 |

**Live day replay (2026-05-11, 24 h, both zones averaged)**: pinball loss 6.35 vs. 13.22 for the
originally submitted (leaky) forecast and 13.64 for the best naive; CQR interval coverage 96% vs. 56%.
Caveat: the TSO forecasts for 11 May were published on 10 May, i.e. after the challenge deadline, so
the replay reflects the standard day-ahead information set rather than the challenge's.

**Cross-zone differences** (gain share of the q=0.45 model): DE-LU is driven by fundamentals —
residual-load forecast (20%) and renewable share (7%) lead, ahead of yesterday's price (9%).
ES leans much more on price persistence — yesterday's same-hour price (38%) — with residual load
second (11%), plausibly because the less interconnected Iberian market's price level is set
by slower-moving drivers (hydro reservoirs, gas) that the features only capture through past prices.

## Project Structure
```
frigg_hack/
├── data/                          # Raw and processed data
│   ├── raw/                       # Original downloaded data
│   │   ├── delu/                  # DE-LU zone data
│   │   └── es/                    # ES zone data
│   ├── processed/                 # Cleaned and preprocessed data
│   └── external/                  # External data sources (weather, fuel prices)
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_acquisition.ipynb
│   ├── 02_eda_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_development.ipynb
│   └── final_submission.ipynb     # Main submission notebook
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── data/                      # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   └── preprocessors.py
│   ├── features/                  # Feature engineering
│   │   ├── __init__.py
│   │   ├── temporal.py
│   │   ├── weather.py
│   │   └── market.py
│   ├── models/                    # Model implementations
│   │   ├── __init__.py
│   │   ├── short_term.py
│   │   ├── medium_term.py
│   │   ├── long_term.py
│   │   └── ensemble.py
│   ├── evaluation/                # Evaluation metrics
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       └── helpers.py
├── models/                        # Saved model artifacts
│   ├── delu/
│   └── es/
├── outputs/                       # Predictions and visualizations
│   └── predictions.csv
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment (optional)
└── README.md                      # This file
```

## Development Timeline (4 Days)

### Day 1: Infrastructure & Data
- [x] Project setup
- [ ] Data acquisition pipeline
- [ ] Initial EDA

### Day 2: Feature Engineering & Short-term Models
- [ ] Feature engineering pipeline
- [ ] Short-term model development
- [ ] Quantile regression implementation

### Day 3: Medium/Long-term & Ensemble
- [ ] Medium/long-term models
- [ ] Horizon-aware routing
- [ ] Cross-zone analysis

### Day 4: Optimization & Submission
- [ ] Fine-tuning for pinball loss
- [ ] Generate predictions
- [ ] Final notebook and documentation

## Key Features

### Model Architecture
- **Short-term (0-7 days)**: Temporal Transformer + XGBoost + LSTM ensemble
- **Medium-term (7 days - 3 months)**: Prophet + Seasonal decomposition
- **Long-term (3+ months)**: Trend extrapolation + Historical volatility

### Feature Categories
1. **Temporal**: Hour, day, week, month, holidays, seasonal cycles
2. **Weather**: Wind speed, solar irradiance, temperature
3. **Market**: Price lags, volatility, generation mix
4. **Fuel**: Gas, coal, carbon prices
5. **Interconnection**: Cross-border flows (DE-LU specific)

### Zone-Specific Considerations
- **DE-LU**: High wind variability, frequent negative prices, strong interconnections
- **ES**: High solar penetration, hydro flexibility, more isolated market

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from src.models.ensemble import EnsembleForecaster

# Initialize forecaster
forecaster = EnsembleForecaster(zone='DE-LU')

# Make predictions
predictions = forecaster.predict(
    start='2026-05-08T18:00:00+01:00',
    end='2026-05-09T23:00:00+01:00'
)
```

## Evaluation Metric

Asymmetric pinball loss at q=0.45:
```python
def scoring_loss(y_true, y_pred, q=0.45):
    r = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.mean(np.where(r >= 0, q * r, (q - 1) * r)))
```

## Data Sources
- Historical DAA prices: https://energy-charts.info/
- Weather data: Open-Meteo API / ERA5 reanalysis
- Fuel prices: ICE/EEX market data
- Generation data: ENTSO-E Transparency Platform

## Team
Developed for the Frigg Hackathon 2026

## License
Non-commercial use only (Hackathon submission)