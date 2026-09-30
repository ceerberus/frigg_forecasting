"""
Leakage-free features for day-ahead electricity price forecasting.

Information set
---------------
The day-ahead auction for delivery day D closes at noon on D-1 (local time). At that
point the forecaster knows:
  * all auction prices up to and including delivery day D-1 (set by the auction on D-2),
  * the TSOs' day-ahead forecasts of load, solar and wind for day D,
  * the calendar.
It does NOT know any price of day D, nor realised generation/load of day D.

Every price-derived feature for a row in delivery day D is therefore looked up from
day D - lead_days or earlier (lead_days=1 is the standard day-ahead setting). The same
function is used for training and for inference, so there is no train/serve skew.
"""

from typing import List

import numpy as np
import pandas as pd

try:
    import holidays as holidays_lib
    _HOLIDAYS_AVAILABLE = True
except ImportError:
    _HOLIDAYS_AVAILABLE = False


TIMEZONE = {'DE-LU': 'Europe/Berlin', 'ES': 'Europe/Madrid'}
HOLIDAY_COUNTRY = {'DE-LU': 'DE', 'ES': 'ES'}
PEAK_HOURS = range(8, 20)

TARGET = 'price_eur_mwh'


def build_features(df: pd.DataFrame, zone: str, lead_days: int = 1) -> pd.DataFrame:
    """
    Build the feature matrix for every hourly row of `df`.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly data with columns ['timestamp' (UTC), 'price_eur_mwh', 'load_fc_mw',
        'solar_fc_mw', 'wind_fc_mw']. Prices of the days being forecast may be NaN.
    zone : str
        'DE-LU' or 'ES' (selects local time zone and holiday calendar).
    lead_days : int
        Days between the last fully known price day and the delivery day (1 = day-ahead).

    Returns
    -------
    pd.DataFrame
        'timestamp', 'delivery_date', the target and all features (see `feature_columns`).
    """
    if not 1 <= lead_days <= 7:
        raise ValueError("lead_days must be between 1 and 7")

    df = df.sort_values('timestamp').reset_index(drop=True)
    local = df['timestamp'].dt.tz_convert(TIMEZONE[zone])
    date = local.dt.tz_localize(None).dt.normalize()
    hour = local.dt.hour
    L = pd.Timedelta(days=lead_days)

    out = pd.DataFrame({'timestamp': df['timestamp'], 'delivery_date': date, TARGET: df[TARGET]})

    # ── Price history: only days <= D - lead_days ───────────────────────────
    hourly = pd.DataFrame({'date': date, 'hour': hour, 'price': df[TARGET]})
    by_hour = hourly.groupby(['date', 'hour'])['price'].mean()       # DST: 25h day -> mean
    all_days = pd.date_range(date.min(), date.max(), freq='D')
    daily = hourly.groupby('date')['price'].agg(['mean', 'min', 'max', 'std', 'last']).reindex(all_days)
    daily['peak'] = hourly[hourly['hour'].isin(PEAK_HOURS)].groupby('date')['price'].mean()
    daily['mean_7d'] = daily['mean'].rolling(7, min_periods=5).mean()
    daily['std_7d'] = hourly.groupby('date')['price'].std().reindex(all_days).rolling(7, min_periods=5).mean()

    def same_hour(days_back: pd.Timedelta) -> np.ndarray:
        return by_hour.reindex(pd.MultiIndex.from_arrays([date - days_back, hour])).values

    def day_stat(col: str, days_back: pd.Timedelta) -> np.ndarray:
        return daily[col].reindex(date - days_back).values

    out['price_same_hour_ref'] = same_hour(L)
    out['price_same_hour_ref_m1'] = same_hour(L + pd.Timedelta(days=1))
    out['price_same_hour_d7'] = same_hour(pd.Timedelta(days=7))
    out['price_ref_day_mean'] = day_stat('mean', L)
    out['price_ref_day_min'] = day_stat('min', L)
    out['price_ref_day_max'] = day_stat('max', L)
    out['price_ref_day_std'] = day_stat('std', L)
    out['price_ref_day_peak'] = day_stat('peak', L)
    out['price_ref_day_last'] = day_stat('last', L)
    out['price_ref_mean_7d'] = day_stat('mean_7d', L)
    out['price_ref_std_7d'] = day_stat('std_7d', L)
    out['price_ref_trend'] = out['price_ref_day_mean'] - day_stat('mean', L + pd.Timedelta(days=1))

    # ── TSO day-ahead forecasts for the delivery hour ───────────────────────
    load, solar, wind = df['load_fc_mw'], df['solar_fc_mw'], df['wind_fc_mw']
    resid = load - solar - wind
    out['load_fc_mw'] = load
    out['solar_fc_mw'] = solar
    out['wind_fc_mw'] = wind
    out['residual_load_fc_mw'] = resid
    out['renewable_share_fc'] = (solar + wind) / load

    # Whole-day shape of the forecast (all hours of D are published before the auction)
    resid_by_day = resid.groupby(date)
    out['residual_load_fc_day_mean'] = resid_by_day.transform('mean')
    out['residual_load_fc_day_max'] = resid_by_day.transform('max')
    out['residual_load_fc_day_min'] = resid_by_day.transform('min')
    out['residual_load_fc_rank_in_day'] = resid_by_day.rank(pct=True)

    # Change in fundamentals vs. the reference day (forecast vs. forecast: both known)
    resid_hourly = pd.Series(resid.values, index=pd.MultiIndex.from_arrays([date, hour])).groupby(level=[0, 1]).mean()
    resid_ref = resid_hourly.reindex(pd.MultiIndex.from_arrays([date - L, hour])).values
    out['residual_load_fc_delta_ref'] = resid - resid_ref
    out['residual_load_fc_day_delta_ref'] = (out['residual_load_fc_day_mean']
                                             - resid.groupby(date).mean().reindex(date - L).values)

    # ── Calendar ────────────────────────────────────────────────────────────
    out['hour'] = hour
    out['day_of_week'] = date.dt.dayofweek
    out['is_weekend'] = (out['day_of_week'] >= 5).astype(int)
    out['month'] = date.dt.month
    doy = date.dt.dayofyear
    out['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    out['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    if _HOLIDAYS_AVAILABLE:
        cal = holidays_lib.country_holidays(HOLIDAY_COUNTRY[zone], years=range(date.dt.year.min(), date.dt.year.max() + 1))
        is_hol = date.dt.date.map(lambda d: d in cal).astype(int)
    else:
        is_hol = pd.Series(0, index=date.index)
    out['is_holiday'] = is_hol
    out['is_non_working_day'] = ((out['is_weekend'] == 1) | (is_hol == 1)).astype(int)

    return out


def feature_columns(features: pd.DataFrame) -> List[str]:
    """All model inputs (everything except identifiers and the target)."""
    return [c for c in features.columns if c not in ('timestamp', 'delivery_date', TARGET)]
