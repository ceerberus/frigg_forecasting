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
        
        print(f"✅ Created {len([c for c in df.columns if c != 'timestamp'])} features")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    # ========================================================================
    # Feature Creation Methods
    # ========================================================================
    
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
        
        # Price rolling stats
        if 'price_eur_mwh' in df.columns:
            for window in [6, 12, 24, 168]:  # 6h, 12h, 24h, 1week
                df[f'price_rolling_mean_{window}h'] = df['price_eur_mwh'].rolling(window, min_periods=1).mean()
                df[f'price_rolling_std_{window}h'] = df['price_eur_mwh'].rolling(window, min_periods=1).std()
                df[f'price_rolling_min_{window}h'] = df['price_eur_mwh'].rolling(window, min_periods=1).min()
                df[f'price_rolling_max_{window}h'] = df['price_eur_mwh'].rolling(window, min_periods=1).max()
        
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
        
        # Price changes
        if 'price_eur_mwh' in df.columns:
            df['price_diff_1h'] = df['price_eur_mwh'].diff(1)
            df['price_diff_24h'] = df['price_eur_mwh'].diff(24)
            df['price_pct_change_1h'] = df['price_eur_mwh'].pct_change(1)
            df['price_pct_change_24h'] = df['price_eur_mwh'].pct_change(24)
        
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

