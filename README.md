# Electricity Price Forecasting — Frigg Hackathon 2026

Ensemble forecasting system for European day-ahead electricity prices (DE-LU and ES zones).

## Evaluation Results — May 11 2026

![Actual vs Predicted](outputs/forecasts/actual_vs_predicted.png)

| Zone | Pinball Loss q=0.45 | MAE |
|------|--------------------:|----:|
| DE-LU | 14.75 EUR/MWh | 28.52 |
| ES | 13.28 EUR/MWh | 24.85 |

### Why the model underperformed

**1. Spark spread leakage during training.**
The most important feature (`spark_spread`, ~38–52% gain) was defined as
`price(t) − gas_cost(t)` — meaning the training target was embedded directly in the
feature. The model achieved low validation loss partly because it could reconstruct
the price from itself. At true inference time, the current price is unknown, so a
stale 7-day-old proxy was substituted. The model relied heavily on a signal it could
never actually observe.

**2. One-day data gap.**
Predictions were submitted on May 9; the evaluation window is May 11. All
short-horizon lag features (`price_lag_1h` … `price_lag_48h`) that would normally
carry yesterday's market state were unavailable. They were filled from a reference
row 7 days prior, effectively making `lag_24h ≈ lag_168h` and blinding the model to
actual market conditions on May 10.

**3. May 11 was atypical.**
The model learned to predict a large evening spike in DE-LU (from gas-driven peak
hours in the training data) and a deep solar trough in ES (duck-curve pattern). On
May 11 neither materialised: DE-LU prices stayed flat around 100–130 EUR/MWh and ES
prices remained elevated through midday. Models that had memorised these patterns
were penalised heavily.

**4. Validation scores were misleading.**
In-sample validation used actual spark_spread values, so the reported scores
(DE-LU 4.10 / ES 0.20) significantly overstated real-world performance. The jump
to 14.75 / 13.28 on the live evaluation reflects the combination of leakage, the
data gap, and distribution shift — not a sudden drop in model quality.

## Project Structure

```
frigg_forecasting/
├── data/
│   ├── raw/                        # Historical DAA prices & generation per zone
│   ├── processed/                  # Merged datasets with initial features
│   ├── engineered/                 # Final feature matrices for training
│   └── external/                   # Weather, fuel prices, ENTSO-E, neighbor gen
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_development.ipynb
│   └── heal_the_grid_model.ipynb   # Main submission notebook
├── src/
│   ├── data/                       # Data loaders (ENTSO-E, fuel, neighbors, RTE)
│   ├── features/                   
│   └── evaluation/                 
├── outputs/
│   ├── models/                     # Saved CatBoost / LightGBM / XGBoost artifacts
│   ├── forecasts/                  
│   └── plots/                      
├── submission/                     # Final submission package
│   ├── heal_the_grid_model.ipynb
│   ├── heal_the_grid_predictions.csv
│   └── heal_the_grid_data.zip
├── pitch_website/
├── prepare_full_dataset.py
├── create_fuel_prices.py           # Synthetic fuel prices
├── create_submission_package.py
└── requirements.txt
```

## Models

Quantile ensemble of CatBoost, LightGBM, and XGBoost trained per zone (DE-LU, ES) at quantiles **p025, p45, p50, p975**.

## Evaluation Metric

Asymmetric pinball loss at q=0.45:

```python
def scoring_loss(y_true, y_pred, q=0.45):
    r = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.mean(np.where(r >= 0, q * r, (q - 1) * r)))
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Data Sources

- DAA prices: [energy-charts.info](https://energy-charts.info/)
- Weather: Open-Meteo API
- Generation forecasts & cross-border flows: ENTSO-E Transparency Platform
- Fuel prices: ICE/EEX market data
- Neighbor generation (FR, PT): ENTSO-E
