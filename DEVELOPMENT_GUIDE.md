# Development Guide - Electricity Price Forecasting

## Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Project Structure Overview

```
frigg_hack/
├── data/                    # All data files
│   ├── raw/                 # Original downloaded data
│   ├── processed/           # Cleaned, merged datasets
│   └── external/            # Weather and auxiliary data
├── notebooks/               # Jupyter notebooks (development)
├── src/                     # Source code modules
│   ├── data/               # Data loading
│   ├── features/           # Feature engineering
│   ├── models/             # Model implementations
│   ├── evaluation/         # Metrics and evaluation
│   └── utils/              # Utilities
├── models/                  # Saved model artifacts
└── outputs/                 # Predictions and visualizations
```

## Development Workflow

### Phase 1: Data Acquisition (Day 1 Morning)
**Goal**: Download and cache all required data

1. Run `notebooks/01_data_acquisition.ipynb`
2. This will download:
   - Historical DAA prices (2020-2026) for DE-LU and ES
   - Generation data by source
   - Weather data (temperature, wind, solar)
3. Data is cached in `data/raw/` and `data/external/`
4. Merged datasets saved to `data/processed/`

**Key Data Sources**:
- Energy-charts.info: Historical prices and generation
- ENTSO-E Transparency Platform: Real-time generation data
- Open-Meteo API: Weather forecasts and historical data
- ERA5 Reanalysis: Historical weather data

### Phase 2: Exploratory Data Analysis (Day 1 Afternoon)
**Goal**: Understand market characteristics and differences between zones

1. Run `notebooks/02_eda_analysis.ipynb`
2. Analyze:
   - Price distributions and volatility
   - Seasonal patterns
   - Negative price occurrences
   - Generation mix differences
   - Weather correlations
3. Document key insights for model design

**Key Questions**:
- How do DE-LU and ES markets differ?
- What drives negative prices?
- Which features correlate most with prices?
- Are there regime changes or structural breaks?

### Phase 3: Feature Engineering (Day 2 Morning)
**Goal**: Create rich feature set for modeling

1. Implement feature engineering in `src/features/`
2. Create features:
   - **Temporal**: Hour, day, week, month, holidays, seasonal cycles
   - **Lags**: Price lags (1h, 24h, 168h), rolling statistics
   - **Weather**: Wind speed, solar irradiance, temperature
   - **Generation**: Renewable penetration, fossil dispatch
   - **Market**: Volatility, price spreads, interconnection flows
3. Ensure same feature vocabulary for both zones

**Feature Engineering Principles**:
- No data leakage (only use past information)
- Handle missing values appropriately
- Scale/normalize features
- Create interaction terms where appropriate

### Phase 4: Model Development (Day 2-3)
**Goal**: Build multi-horizon forecasting system

#### Short-term Model (0-7 days)
- **Architecture**: Ensemble of Transformer + XGBoost + LSTM
- **Features**: All available features (rich, granular)
- **Training**: Quantile regression for p025, p50, p975
- **Loss**: Asymmetric pinball loss (q=0.45)

#### Medium-term Model (7 days - 3 months)
- **Architecture**: Prophet + Seasonal decomposition
- **Features**: Seasonal patterns, trends, exogenous variables
- **Training**: Probabilistic forecasting with uncertainty

#### Long-term Model (3+ months)
- **Architecture**: Trend extrapolation + Historical volatility
- **Features**: Long-term trends, structural factors
- **Uncertainty**: Historical volatility-based intervals

#### Horizon Router
- Automatically select appropriate model based on forecast horizon
- Smooth transitions between models
- Ensemble weighting based on horizon

### Phase 5: Training & Validation (Day 3)
**Goal**: Train models and validate performance

1. Implement walk-forward validation
2. Train separate models for DE-LU and ES
3. Tune hyperparameters
4. Validate on held-out periods
5. Analyze feature importance differences

**Validation Strategy**:
- Time series cross-validation (no random splits!)
- Multiple validation periods (different seasons, market conditions)
- Test on crisis periods and normal periods
- Validate uncertainty calibration

### Phase 6: Final Predictions (Day 4)
**Goal**: Generate submission and documentation

1. Generate predictions for evaluation window:
   - Start: 2026-05-08 18:00 CEST (17:00 UTC)
   - End: 2026-05-09 23:00 CEST (22:00 UTC)
2. Create final submission notebook
3. Write methodology documentation
4. Prepare data zip with README

## Key Implementation Details

### Asymmetric Loss Function
```python
def pinball_loss(y_true, y_pred, q=0.45):
    """
    Overestimation penalty: 1.22x underestimation penalty
    """
    r = y_true - y_pred
    return np.mean(np.where(r >= 0, q * r, (q - 1) * r))
```

### Quantile Regression
- Train three models: p025, p50, p975
- Or use single model with quantile loss
- Ensure p025 < p50 < p975 (monotonicity)

### Zone-Specific Considerations

**DE-LU**:
- High wind variability → wind forecasts critical
- Frequent negative prices → special handling needed
- Strong interconnections → neighbor prices matter
- Winter heating peaks → temperature sensitivity

**ES**:
- High solar penetration → solar irradiance dominant
- Hydro flexibility → reservoir levels important
- More isolated → less cross-border influence
- Summer cooling peaks → AC demand modeling

### Model Evaluation
```python
from src.evaluation import evaluate_forecast

metrics = evaluate_forecast(
    y_true=actual_prices,
    y_pred_p50=predictions_median,
    y_pred_p025=predictions_lower,
    y_pred_p975=predictions_upper
)
```

## Tips for Winning

### 1. Feature Engineering Excellence
- Create domain-specific features (not just generic time series features)
- Capture market microstructure (bidding behavior, merit order)
- Use weather forecasts appropriately (don't overfit to historical weather)

### 2. Probabilistic Forecasting
- Properly calibrated uncertainty is as important as point forecasts
- Use conformal prediction for calibration
- Validate coverage empirically

### 3. Asymmetric Loss Optimization
- Train specifically for q=0.45 pinball loss
- Bias predictions slightly lower to avoid overestimation penalty
- This is a key differentiator!

### 4. Cross-Zone Analysis
- Explain WHY models differ between zones
- Show feature importance differences
- Demonstrate understanding of market structure

### 5. Documentation Quality
- Clear methodology explanation
- Insightful visualizations
- Professional presentation
- Reproducible code

## Common Pitfalls to Avoid

1. **Data Leakage**: Never use future information in features
2. **Overfitting**: Strong regularization, ensemble diversity
3. **Ignoring Uncertainty**: Point forecasts alone are insufficient
4. **Wrong Validation**: Must use time series splits, not random
5. **Feature Mismatch**: Same feature vocabulary for both zones
6. **Computational Issues**: Cache preprocessed data, use efficient code

## Debugging Tips

### Data Issues
```python
# Check for missing values
df.isnull().sum()

# Check timestamp continuity
df['timestamp'].diff().value_counts()

# Check for duplicates
df['timestamp'].duplicated().sum()
```

### Model Issues
```python
# Check predictions are reasonable
predictions.describe()

# Check monotonicity of quantiles
assert (predictions_p025 <= predictions_p50).all()
assert (predictions_p50 <= predictions_p975).all()

# Validate on known period
evaluate_forecast(y_true, y_pred_p50, y_pred_p025, y_pred_p975)
```

## Resources

### Data Sources
- https://energy-charts.info/ - Historical prices and generation
- https://transparency.entsoe.eu/ - ENTSO-E data
- https://open-meteo.com/ - Weather data API
- https://www.eex.com/ - Fuel prices

### Documentation
- Challenge specification (see task description)
- Pinball loss: https://en.wikipedia.org/wiki/Quantile_regression
- Prophet: https://facebook.github.io/prophet/
- XGBoost quantile: https://xgboost.readthedocs.io/

## Next Steps

1. ✅ Project structure created
2. ✅ Core modules implemented (metrics, loaders)
3. ✅ Data acquisition notebook ready
4. ⏳ Run data acquisition
5. ⏳ Implement feature engineering
6. ⏳ Build models
7. ⏳ Train and validate
8. ⏳ Generate predictions
9. ⏳ Create submission

**Current Status**: Infrastructure ready, ready to start data acquisition!

Good luck! 🚀