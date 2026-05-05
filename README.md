# Electricity Price Forecasting Challenge

## Project Overview
Multi-horizon ensemble forecasting system for European electricity markets (DE-LU and ES zones).

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