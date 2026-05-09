"""
Feature Engineering for Electricity Price Forecasting

This module creates powerful features that capture:
- Temporal patterns (hour, day, week, season)
- Price dynamics (lags, rolling stats, volatility)
- Supply-demand balance (renewable share, residual load)
- Market interactions (generation mix, weather correlations)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import warnings

try:
    import holidays as holidays_lib
    _HOLIDAYS_AVAILABLE = True
except ImportError:
    _HOLIDAYS_AVAILABLE = False


class FeatureEngineer:
    """
    Transform raw electricity market data into ML-ready features.
    
    Key Feature Categories:
    1. Lag Features - Historical values (1h, 24h, 168h lags)
    2. Rolling Statistics - Moving averages, std, min, max
    3. Temporal Features - Hour, day, week, month, holidays
    4. Interactions - Feature combinations (renewable × load, etc.)
    5. Differentials - Rate of change features
    """
    
    def __init__(self, zone: str = 'DE-LU'):
        """
        Initialize Feature Engineer for a specific bidding zone.
        
        Parameters
        ----------
        zone : str
            Bidding zone ('DE-LU' or 'ES')
        """
        self.zone = zone
        self.feature_names_ = None
        
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Fit the feature engineer (learns feature names).
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data with timestamp and raw features
            
        Returns
        -------
        self
        """
        # Just store feature names for consistency
        df_features = self.transform(df.copy())
        self.feature_names_ = [col for col in df_features.columns if col != 'timestamp']
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw data into engineered features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw data with timestamp, price, generation, weather
            
        Returns
        -------
        pd.DataFrame
            Data with engineered features
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"🔧 Engineering features for {self.zone}...")

        # 0. Derive base aggregates before lag/rolling steps
        df = self._derive_base_columns(df)

        # 1. Temporal Features
        df = self._add_temporal_features(df)
        
        # 2. Lag Features (critical for time series!)
        df = self._add_lag_features(df)
        
        # 3. Rolling Statistics
        df = self._add_rolling_features(df)
        
        # 4. Differential Features
        df = self._add_differential_features(df)
        
        # 5. Interaction Features
        df = self._add_interaction_features(df)
        
        # 6. Cyclical Encoding (for hour, day, month)
        df = self._add_cyclical_features(df)

        # 7. Fuel price features (gas, carbon, marginal costs)
        df = self._add_fuel_price_features(df)

        # 8. ENTSO-E features (cross-border flows, nuclear, hydro, pumped storage, forecasts)
        df = self._add_entsoe_features(df)

        # 9. Neighbor country features (French nuclear, Portuguese hydro, etc.)
        df = self._add_neighbor_features(df)

        print(f"✅ Created {len([c for c in df.columns if c != 'timestamp'])} features")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    # ========================================================================
    # Feature Creation Methods
    # ========================================================================
    
    def _derive_base_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive aggregate columns that must exist before lag/rolling steps.
        Called first so total_load_mw, total_renewable_mw, residual_load_mw
        are available when _add_lag_features runs.
        """
        # total_renewable_mw — prefer ENTSO-E actuals, fall back to Energy-Charts cols
        if "total_renewable_mw" not in df.columns:
            ren_candidates = [
                "wind_onshore_entsoe_mw", "wind_offshore_entsoe_mw", "solar_entsoe_mw",
                "wind_onshore", "wind_offshore", "solar",
            ]
            ren_cols = [c for c in ren_candidates if c in df.columns]
            if ren_cols:
                df["total_renewable_mw"] = df[ren_cols].clip(lower=0).sum(axis=1)

        # residual_load_mw = total_load_mw - renewables (key merit-order price driver)
        if "residual_load_mw" not in df.columns and "total_load_mw" in df.columns:
            ren = df["total_renewable_mw"] if "total_renewable_mw" in df.columns else 0
            df["residual_load_mw"] = df["total_load_mw"] - ren

        # renewable_share_pct
        if "renewable_share_pct" not in df.columns and "total_renewable_mw" in df.columns \
                and "total_load_mw" in df.columns:
            df["renewable_share_pct"] = (
                df["total_renewable_mw"] / (df["total_load_mw"] + 1e-6) * 100
            ).clip(0, 200)

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        print("  → Temporal features...")
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['year'] = df['timestamp'].dt.year
        
        # Binary indicators
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = df['hour'].isin(range(0, 6)).astype(int)
        df['is_peak_hour'] = df['hour'].isin([8, 9, 10, 17, 18, 19, 20]).astype(int)

        # Season (meteorological)
        df['season'] = df['month'] % 12 // 3 + 1  # 1=Winter, 2=Spring, 3=Summer, 4=Fall

        # Public holidays — reduce price on non-working days
        if _HOLIDAYS_AVAILABLE:
            country_holidays = holidays_lib.Germany() if self.zone == 'DE-LU' else holidays_lib.Spain()
            df['is_holiday'] = df['timestamp'].dt.date.map(
                lambda d: int(d in country_holidays)
            ).astype(int)
        else:
            df['is_holiday'] = 0

        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged values of key variables."""
        print("  → Lag features...")
        
        # Price lags (most important!)
        if 'price_eur_mwh' in df.columns:
            for lag in [1, 2, 3, 6, 12, 24, 48, 168, 336, 504]:  # up to 3 weeks
                df[f'price_lag_{lag}h'] = df['price_eur_mwh'].shift(lag)

        # Load lags
        if 'total_load_mw' in df.columns:
            for lag in [1, 24, 168, 336]:
                df[f'load_lag_{lag}h'] = df['total_load_mw'].shift(lag)

        # Renewable generation lags
        if 'total_renewable_mw' in df.columns:
            for lag in [1, 24, 168]:
                df[f'renewable_lag_{lag}h'] = df['total_renewable_mw'].shift(lag)

        # Residual load lags
        if 'residual_load_mw' in df.columns:
            for lag in [1, 24, 168, 336]:
                df[f'residual_load_lag_{lag}h'] = df['residual_load_mw'].shift(lag)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics."""
        print("  → Rolling statistics...")
        
        # Price rolling stats over past window (includes lag-1 price, not current)
        if 'price_eur_mwh' in df.columns:
            for window in [6, 12, 24, 168]:
                df[f'price_rolling_mean_{window}h'] = df['price_eur_mwh'].rolling(window, min_periods=1).mean()
                df[f'price_rolling_std_{window}h']  = df['price_eur_mwh'].rolling(window, min_periods=1).std()
                df[f'price_rolling_min_{window}h']  = df['price_eur_mwh'].rolling(window, min_periods=1).min()
                df[f'price_rolling_max_{window}h']  = df['price_eur_mwh'].rolling(window, min_periods=1).max()
        
        # Load rolling stats
        if 'total_load_mw' in df.columns:
            for window in [24, 168]:
                df[f'load_rolling_mean_{window}h'] = df['total_load_mw'].rolling(window, min_periods=1).mean()
        
        # Renewable share rolling stats
        if 'renewable_share_pct' in df.columns:
            for window in [24, 168]:
                df[f'renewable_share_rolling_mean_{window}h'] = df['renewable_share_pct'].rolling(window, min_periods=1).mean()
        
        return df
    
    def _add_differential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rate-of-change features."""
        print("  → Differential features...")
        
        # Price momentum — shift(1) so diff uses price(t-1)-price(t-2), not price(t)
        if 'price_eur_mwh' in df.columns:
            df['price_diff_1h']       = df['price_eur_mwh'].diff(1).shift(1)
            df['price_diff_24h']      = df['price_eur_mwh'].diff(24).shift(24)
            df['price_pct_change_1h'] = df['price_eur_mwh'].pct_change(1).shift(1)
            df['price_pct_change_24h']= df['price_eur_mwh'].pct_change(24).shift(24)
        
        # Load changes
        if 'total_load_mw' in df.columns:
            df['load_diff_1h'] = df['total_load_mw'].diff(1)
            df['load_diff_24h'] = df['total_load_mw'].diff(24)
        
        # Renewable changes
        if 'total_renewable_mw' in df.columns:
            df['renewable_diff_1h'] = df['total_renewable_mw'].diff(1)
            df['renewable_diff_24h'] = df['total_renewable_mw'].diff(24)
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature interactions."""
        print("  → Interaction features...")
        
        # Renewable share × Load (high renewable at high load = low prices)
        if 'renewable_share_pct' in df.columns and 'total_load_mw' in df.columns:
            df['renewable_load_interaction'] = df['renewable_share_pct'] * df['total_load_mw'] / 100
        
        # Residual load ratio (key price driver!)
        if 'residual_load_mw' in df.columns and 'total_load_mw' in df.columns:
            df['residual_load_ratio'] = df['residual_load_mw'] / (df['total_load_mw'] + 1e-6)
        
        # Wind + Solar combined
        if 'wind_onshore' in df.columns and 'solar' in df.columns:
            df['wind_solar_combined'] = df.get('wind_onshore', 0) + df.get('wind_offshore', 0) + df['solar']
        
        # Temperature × Hour (heating/cooling patterns)
        if 'temperature_2m_c' in df.columns and 'hour' in df.columns:
            df['temp_hour_interaction'] = df['temperature_2m_c'] * df['hour']
        
        # HDD × Load (heating demand)
        if 'hdd' in df.columns and 'total_load_mw' in df.columns:
            df['hdd_load_interaction'] = df['hdd'] * df['total_load_mw']
        
        # CDD × Load (cooling demand)
        if 'cdd' in df.columns and 'total_load_mw' in df.columns:
            df['cdd_load_interaction'] = df['cdd'] * df['total_load_mw']
        
        return df
    
    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode cyclical features (hour, day, month) as sin/cos."""
        print("  → Cyclical encoding...")
        
        # Hour (24-hour cycle)
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week (7-day cycle)
        if 'day_of_week' in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month (12-month cycle)
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _add_fuel_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag and rolling features for fuel/carbon prices."""
        print("  → Fuel price features...")

        FUEL_COLS = [
            'gas_ttf_eur_mwh',
            'co2_eua_eur_ton',
            'oil_brent_usd_barrel',
            'gas_marginal_cost_eur_mwh',
            'coal_marginal_cost_eur_mwh',
            'gas_coal_spread_eur_mwh',
        ]

        for col in FUEL_COLS:
            if col not in df.columns:
                continue
            # Lags: 24h (yesterday), 168h (last week), 720h (~last month)
            for lag in [24, 168, 720]:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
            # Rolling means: 7-day and 30-day
            df[f'{col}_roll_7d']  = df[col].rolling(168,  min_periods=1).mean()
            df[f'{col}_roll_30d'] = df[col].rolling(720,  min_periods=1).mean()
            # Rate of change vs yesterday
            df[f'{col}_chg_24h']  = df[col].pct_change(24)

        # Spark spread lagged — yesterday's spread tells us the market regime.
        # Current spread (price(t) - cost(t)) would be pure target leakage.
        if 'price_eur_mwh' in df.columns and 'gas_marginal_cost_eur_mwh' in df.columns:
            spread = df['price_eur_mwh'] - df['gas_marginal_cost_eur_mwh']
            df['spark_spread_lag_24h']  = spread.shift(24)
            df['spark_spread_lag_168h'] = spread.shift(168)

        return df

    def _add_neighbor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lag/rolling features for neighbor-country generation columns.

        For DE-LU: French nuclear (fr_nuclear_mw) is the single most impactful
        external variable — high French nuclear → surplus exports → DE prices fall.

        For ES: Portuguese hydro (pt_hydro_mw) affects Iberian supply balance.
        French nuclear (fr_nuclear_mw) affects ES via FR→ES interconnector.
        """
        print("  → Neighbor country features...")

        # ── French nuclear ──────────────────────────────────────────────────
        if "fr_nuclear" in df.columns:
            # The column may come in as 'fr_nuclear' or 'fr_nuclear_mw'
            col = "fr_nuclear"
        elif "fr_nuclear_mw" in df.columns:
            col = "fr_nuclear_mw"
        else:
            col = None

        if col:
            for lag in [1, 24, 168]:
                df[f"fr_nuclear_lag_{lag}h"] = df[col].shift(lag)
            df["fr_nuclear_roll_24h"]  = df[col].rolling(24,  min_periods=1).mean()
            df["fr_nuclear_roll_168h"] = df[col].rolling(168, min_periods=1).mean()
            df["fr_nuclear_diff_24h"]  = df[col].diff(24)

        # ── French hydro ─────────────────────────────────────────────────────
        fr_hydro_cols = [c for c in df.columns
                         if c.startswith("fr_hydro") or c.startswith("fr_hydro_water")]
        if fr_hydro_cols:
            df["fr_hydro_mw"] = df[fr_hydro_cols].sum(axis=1)
            for lag in [24, 168]:
                df[f"fr_hydro_lag_{lag}h"] = df["fr_hydro_mw"].shift(lag)

        # ── Portuguese hydro (ES-relevant) ───────────────────────────────────
        pt_hydro_cols = [c for c in df.columns
                         if c.startswith("pt_hydro") or c.startswith("pt_hydro_water")]
        if pt_hydro_cols:
            df["pt_hydro_mw"] = df[pt_hydro_cols].sum(axis=1)
            for lag in [1, 24, 168]:
                df[f"pt_hydro_lag_{lag}h"] = df["pt_hydro_mw"].shift(lag)
            df["pt_hydro_roll_168h"] = df["pt_hydro_mw"].rolling(168, min_periods=1).mean()

        # ── French gas (marginal cost signal) ────────────────────────────────
        fr_gas_col = next((c for c in df.columns if "fr_fossil_gas" in c), None)
        if fr_gas_col:
            df["fr_gas_lag_24h"] = df[fr_gas_col].shift(24)

        # ── Combined neighbor low-carbon proxy ───────────────────────────────
        # When FR nuclear + hydro is high, it exports cheap power → suppresses DE-LU/ES prices
        low_c_neighbor = [c for c in df.columns
                          if any(x in c for x in ["fr_nuclear", "fr_hydro", "pt_hydro"])
                          and "lag" not in c and "roll" not in c and "diff" not in c]
        if low_c_neighbor:
            df["neighbor_low_carbon_mw"] = df[low_c_neighbor].sum(axis=1)
            df["neighbor_low_carbon_lag_24h"] = df["neighbor_low_carbon_mw"].shift(24)

        # ── RTE D-1 nuclear forecast ──────────────────────────────────────────
        # The day-ahead nuclear forecast published by RTE at noon D-1.
        # Unlike the actual generation lags, this is genuinely available at
        # inference time for the next day — no train/inference mismatch.
        if "fr_nuclear_forecast_mw" in df.columns:
            fc = df["fr_nuclear_forecast_mw"]
            df["fr_nuclear_forecast_lag_24h"]  = fc.shift(24)
            df["fr_nuclear_forecast_roll_24h"] = fc.rolling(24,  min_periods=1).mean()
            # Forecast error vs actual (only meaningful in training — zero at inference)
            if col:  # col = actual fr_nuclear column defined above
                df["fr_nuclear_forecast_error"] = df[col] - fc
                df["fr_nuclear_forecast_err_24h"] = df["fr_nuclear_forecast_error"].shift(24)

        return df

    def _add_entsoe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag/rolling features for ENTSO-E cross-border flows and generation mix."""
        print("  → ENTSO-E features...")

        # ── Cross-border flows ───────────────────────────────────────────────
        # French imports are the single most impactful external driver for DE-LU:
        # high FR nuclear → large FR→DE exports → suppresses German prices
        flow_cols = [c for c in df.columns if c.startswith("net_import_") and c != "net_import_total_mw"]
        for col in flow_cols + ["net_import_total_mw"]:
            if col not in df.columns:
                continue
            for lag in [1, 24, 168]:
                df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
            df[f"{col}_roll_24h"] = df[col].rolling(24, min_periods=1).mean()

        # ── Nuclear generation ───────────────────────────────────────────────
        if "nuclear_mw" in df.columns:
            for lag in [1, 24, 168]:
                df[f"nuclear_lag_{lag}h"] = df["nuclear_mw"].shift(lag)
            df["nuclear_roll_24h"]  = df["nuclear_mw"].rolling(24,  min_periods=1).mean()
            df["nuclear_roll_168h"] = df["nuclear_mw"].rolling(168, min_periods=1).mean()

        # ── Pumped storage net (market equilibrium signal) ───────────────────
        # Positive = generating (high price expected)
        # Negative = pumping (low / negative price expected)
        if "pumped_storage_net_mw" in df.columns:
            for lag in [1, 24]:
                df[f"pumped_net_lag_{lag}h"] = df["pumped_storage_net_mw"].shift(lag)
            df["pumped_net_roll_24h"] = df["pumped_storage_net_mw"].rolling(24, min_periods=1).mean()
            df["is_pumping"] = (df["pumped_storage_net_mw"] < 0).astype(int)

        # ── Hydro (key price driver for ES) ─────────────────────────────────
        if "hydro_total_mw" in df.columns:
            for lag in [1, 24, 168]:
                df[f"hydro_lag_{lag}h"] = df["hydro_total_mw"].shift(lag)
            df["hydro_roll_24h"]  = df["hydro_total_mw"].rolling(24,  min_periods=1).mean()
            df["hydro_roll_168h"] = df["hydro_total_mw"].rolling(168, min_periods=1).mean()
            df["hydro_diff_24h"]  = df["hydro_total_mw"].diff(24)

        # ── Fossil mix ───────────────────────────────────────────────────────
        if "fossil_mix_mw" in df.columns:
            for lag in [1, 24]:
                df[f"fossil_mix_lag_{lag}h"] = df["fossil_mix_mw"].shift(lag)

        # ── Day-ahead generation forecast (forward-looking market signal) ────
        for col in ["wind_forecast_mw", "solar_forecast_mw", "renewable_forecast_total_mw"]:
            if col not in df.columns:
                continue
            for lag in [24, 168]:
                df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
            df[f"{col}_roll_24h"] = df[col].rolling(24, min_periods=1).mean()

        # Forecast vs actual (surprise = when renewables exceed/miss forecast)
        if "wind_forecast_mw" in df.columns and "wind_onshore_entsoe_mw" in df.columns:
            df["wind_forecast_error"] = (
                df["wind_onshore_entsoe_mw"] - df["wind_forecast_mw"]
            )
        if "solar_forecast_mw" in df.columns and "solar_entsoe_mw" in df.columns:
            df["solar_forecast_error"] = (
                df["solar_entsoe_mw"] - df["solar_forecast_mw"]
            )

        # ── Day-ahead load forecast (forward-looking demand signal) ─────────
        # Published D-1 by TSOs — available at inference time, no leakage.
        if "load_forecast_mw" in df.columns:
            for lag in [24, 168]:
                df[f"load_forecast_lag_{lag}h"] = df["load_forecast_mw"].shift(lag)
            df["load_forecast_roll_24h"] = df["load_forecast_mw"].rolling(24, min_periods=1).mean()

        # ── Low-carbon share (from ENTSO-E generation mix) ──────────────────
        low_c_cols = [c for c in ["nuclear_mw", "hydro_total_mw",
                                   "wind_onshore_entsoe_mw", "wind_offshore_entsoe_mw",
                                   "solar_entsoe_mw", "biomass_mw", "other_renewable_mw"]
                      if c in df.columns]
        all_gen_cols = [c for c in df.columns
                        if c.endswith("_mw") and not c.startswith("net_import")
                        and not c.startswith("pumped_storage_net")
                        and not c.startswith("wind_forecast")
                        and not c.startswith("solar_forecast")
                        and not c.startswith("renewable_forecast")]
        if low_c_cols and all_gen_cols:
            low_c = df[low_c_cols].clip(lower=0).sum(axis=1)
            total = df[all_gen_cols].clip(lower=0).sum(axis=1)
            df["low_carbon_share_entsoe"] = low_c / (total + 1e-6)

        return df

    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_feature_names(self) -> List[str]:
        """Get list of all engineered feature names."""
        if self.feature_names_ is None:
            raise ValueError("FeatureEngineer must be fitted first!")
        return self.feature_names_
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Group features by category for analysis."""
        if self.feature_names_ is None:
            raise ValueError("FeatureEngineer must be fitted first!")
        
        groups = {
            'temporal': [],
            'lag': [],
            'rolling': [],
            'differential': [],
            'interaction': [],
            'cyclical': [],
            'raw': []
        }
        
        for feat in self.feature_names_:
            if 'lag' in feat:
                groups['lag'].append(feat)
            elif 'rolling' in feat:
                groups['rolling'].append(feat)
            elif 'diff' in feat or 'pct_change' in feat:
                groups['differential'].append(feat)
            elif 'interaction' in feat or 'combined' in feat:
                groups['interaction'].append(feat)
            elif 'sin' in feat or 'cos' in feat:
                groups['cyclical'].append(feat)
            elif any(x in feat for x in ['hour', 'day', 'week', 'month', 'season', 'is_']):
                groups['temporal'].append(feat)
            else:
                groups['raw'].append(feat)
        
        return groups


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str = 'price_eur_mwh',
    drop_na: bool = True
) -> tuple:
    """
    Prepare data for model training.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered data
    target_col : str
        Target variable column name
    drop_na : bool
        Whether to drop rows with NaN values
        
    Returns
    -------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    """
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['timestamp', target_col]]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    if drop_na:
        # Drop rows where target is NaN
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Drop rows where any feature is NaN
        valid_idx = ~X.isna().any(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
    
    print(f"✅ Training data prepared: {len(X)} samples, {len(feature_cols)} features")
    
    return X, y

