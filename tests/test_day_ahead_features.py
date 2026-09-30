"""Guards against target leakage in the day-ahead feature pipeline."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.day_ahead import build_features, feature_columns, TIMEZONE  # noqa: E402


def _synthetic(n_days: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range('2025-03-01', periods=24 * n_days, freq='1h', tz='UTC')  # spans a DST switch
    return pd.DataFrame({
        'timestamp': ts,
        'price_eur_mwh': rng.normal(80, 30, len(ts)),
        'load_fc_mw': rng.normal(50_000, 5_000, len(ts)),
        'solar_fc_mw': rng.uniform(0, 20_000, len(ts)),
        'wind_fc_mw': rng.uniform(0, 30_000, len(ts)),
    })


@pytest.mark.parametrize('zone', ['DE-LU', 'ES'])
@pytest.mark.parametrize('lead_days', [1, 2])
def test_features_of_day_d_ignore_prices_from_day_d_minus_lead_plus_one_onwards(zone, lead_days):
    df = _synthetic()
    local_date = df['timestamp'].dt.tz_convert(TIMEZONE[zone]).dt.tz_localize(None).dt.normalize()
    day_d = pd.Timestamp('2025-03-30')                     # DST switch day in both zones
    first_unknown_day = day_d - pd.Timedelta(days=lead_days - 1)

    perturbed = df.copy()
    unknown = local_date >= first_unknown_day
    perturbed.loc[unknown, 'price_eur_mwh'] = np.random.default_rng(1).normal(500, 200, unknown.sum())

    rows = (local_date == day_d).values
    a = build_features(df, zone, lead_days)
    b = build_features(perturbed, zone, lead_days)
    cols = feature_columns(a)
    pd.testing.assert_frame_equal(a.loc[rows, cols], b.loc[rows, cols])


def test_target_is_not_a_feature():
    feats = build_features(_synthetic(), 'DE-LU')
    assert 'price_eur_mwh' not in feature_columns(feats)
