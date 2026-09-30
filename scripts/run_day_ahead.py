"""
Leakage-free day-ahead pipeline: data -> features -> tuning -> rolling-origin backtest
-> evaluation (incl. Diebold-Mariano tests) -> replay of the live evaluation day.

Timeline (no overlap, so neither tuning nor calibration can leak into the evaluation):
    2023-01-01 .. 2024-10-31   tuning: training data
    2024-11-01 .. 2025-02-28   tuning: validation (Optuna, early stopping)
    2025-03-01 .. 2025-05-10   backtest warm-up (fills the conformal calibration window)
    2025-05-11 .. 2026-05-10   evaluation: 365 days, every day forecast out-of-sample
    2026-05-11                 live evaluation day of the challenge (replayed)

Usage:  python scripts/run_day_ahead.py [--retune]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loaders import EnergyChartsLoader  # noqa: E402
from src.evaluation.metrics import pinball_loss, interval_score, diebold_mariano  # noqa: E402
from src.features.day_ahead import build_features, feature_columns, TARGET  # noqa: E402
from src.models.quantile_lgbm import (  # noqa: E402
    fit_quantile_models, predict_quantiles, rolling_origin_backtest, tune_params,
    cqr_margin, QUANTILES,
)

ZONES = ['DE-LU', 'ES']
DATA_START, DATA_END = '2023-01-01', '2026-05-12'
TUNE_TRAIN_END, TUNE_VALID_END = '2024-11-01', '2025-03-01'
BACKTEST_START, EVAL_START, EVAL_END = '2025-03-01', '2025-05-11', '2026-05-10'
LIVE_DAY_UTC = pd.date_range('2026-05-11 00:00', '2026-05-11 23:00', freq='1h', tz='UTC')
LAST_KNOWN_DAY = pd.Timestamp('2026-05-10')   # prices known at the submission deadline (9 May, 23:59)

OUT = ROOT / 'outputs'
PARAMS_FILE = OUT / 'models' / 'day_ahead_params.json'


def load_zone(loader: EnergyChartsLoader, zone: str) -> pd.DataFrame:
    prices = loader.load_hourly_prices(zone, DATA_START, DATA_END)
    fc = loader.load_power_forecasts(zone, DATA_START, DATA_END)
    for d in (prices, fc):
        d['timestamp'] = pd.to_datetime(d['timestamp'], utc=True)
    return prices.merge(fc, on='timestamp', how='inner')


def evaluate(bt: pd.DataFrame) -> dict:
    """Metrics on the evaluation window, model vs. naive benchmarks."""
    e = bt[(bt['delivery_date'] >= EVAL_START) & (bt['delivery_date'] <= EVAL_END)].dropna(
        subset=['y', 'p45', 'naive_d1', 'naive_d7', 'p025_cqr'])
    y = e['y'].values

    def daily_pinball(col):
        return e.assign(l=np.where(e['y'] >= e[col], 0.45 * (e['y'] - e[col]), 0.55 * (e[col] - e['y']))) \
                .groupby('delivery_date')['l'].mean().values

    m = {'n_hours': int(len(e)), 'n_days': int(e['delivery_date'].nunique())}
    for name, col in [('model', 'p45'), ('naive_d1', 'naive_d1'), ('naive_d7', 'naive_d7')]:
        m[f'pinball_q45_{name}'] = pinball_loss(y, e[col])
        m[f'mae_{name}'] = float(np.mean(np.abs(y - e[col])))
    m['rmse_model'] = float(np.sqrt(np.mean((y - e['p45']) ** 2)))
    best_naive = min(('naive_d1', 'naive_d7'), key=lambda k: m[f'pinball_q45_{k}'])
    m['best_naive'] = best_naive
    m['skill_vs_best_naive_pct'] = 100 * (1 - m['pinball_q45_model'] / m[f'pinball_q45_{best_naive}'])
    m['rmae_vs_naive_d7'] = m['mae_model'] / m['mae_naive_d7']
    for naive in ('naive_d1', 'naive_d7'):
        stat, p = diebold_mariano(daily_pinball('p45'), daily_pinball(naive))
        m[f'dm_stat_vs_{naive}'], m[f'dm_pvalue_vs_{naive}'] = stat, p

    for kind, lo, hi in [('raw', 'p025', 'p975'), ('cqr', 'p025_cqr', 'p975_cqr')]:
        m[f'coverage_95_{kind}'] = float(np.mean((y >= e[lo]) & (y <= e[hi])))
        m[f'mean_width_{kind}'] = float(np.mean(e[hi] - e[lo]))
        m[f'interval_score_{kind}'] = interval_score(y, e[lo], e[hi])
    neg = y < 0
    m['share_negative_price_hours'] = float(neg.mean())
    if neg.any():
        m['coverage_95_cqr_negative_hours'] = float(np.mean((y[neg] >= e['p025_cqr'][neg]) & (y[neg] <= e['p975_cqr'][neg])))
    return m


def replay_live_day(raw: pd.DataFrame, zone: str, cfg: dict, bt: pd.DataFrame) -> pd.DataFrame:
    """
    Re-run the challenge's live forecast honestly: only prices up to local 10 May are used
    (known at the 9 May deadline). UTC 11 May 00-23h = local 11 May 02:00 .. 12 May 01:00,
    so the last two hours are two days ahead (lead_days=2).
    """
    masked = raw.copy()
    local_date = masked['timestamp'].dt.tz_convert('Europe/Berlin').dt.tz_localize(None).dt.normalize()
    actual = masked.set_index('timestamp')[TARGET].copy()
    masked.loc[local_date > LAST_KNOWN_DAY, TARGET] = np.nan

    margin = cqr_margin(bt.loc[bt['delivery_date'] > LAST_KNOWN_DAY - pd.Timedelta(days=90), 'score'].values)
    parts = []
    for lead in (1, 2):
        feats = build_features(masked, zone, lead_days=lead)
        cols = feature_columns(feats)
        train = feats[(feats['delivery_date'] <= LAST_KNOWN_DAY) & feats[TARGET].notna()]
        models = fit_quantile_models(train[cols], train[TARGET], cfg['params'], cfg['num_boost_round'])
        target = feats[feats['timestamp'].isin(LIVE_DAY_UTC)
                       & (feats['delivery_date'] == LAST_KNOWN_DAY + pd.Timedelta(days=lead))]
        p = predict_quantiles(models, target[cols])
        p['timestamp'] = pd.DatetimeIndex(target['timestamp'])
        p['lead_days'] = lead
        p['naive_d1'] = target['price_same_hour_ref'].values
        p['naive_d7'] = target['price_same_hour_d7'].values
        parts.append(p)
    out = pd.concat(parts).sort_values('timestamp').reset_index(drop=True)
    out['p025_cqr'] = out['p025'] - margin
    out['p975_cqr'] = out['p975'] + margin
    out['y'] = actual.reindex(out['timestamp']).values
    return out


def main(retune: bool, reuse_backtest: bool):
    loader = EnergyChartsLoader(cache_dir=ROOT / 'data' / 'raw')
    params = json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() and not retune else {}
    results, replays, importances = {}, [], {}
    (OUT / 'backtest').mkdir(parents=True, exist_ok=True)

    for zone in ZONES:
        print(f'\n=== {zone} ===', flush=True)
        raw = load_zone(loader, zone)
        feats = build_features(raw, zone, lead_days=1)
        cols = feature_columns(feats)
        print(f'{len(feats):,} hourly rows, {len(cols)} features', flush=True)

        if zone not in params:
            print('Tuning (Optuna/TPE, 30 trials)...', flush=True)
            params[zone] = tune_params(feats, cols, TUNE_TRAIN_END, TUNE_VALID_END, n_trials=30)
            PARAMS_FILE.write_text(json.dumps(params, indent=2))
        cfg = params[zone]

        bt_file = OUT / 'backtest' / f'{zone.lower()}_backtest.csv'
        if reuse_backtest and bt_file.exists():
            bt = pd.read_csv(bt_file, parse_dates=['timestamp', 'delivery_date'])
        else:
            print('Rolling-origin backtest...', flush=True)
            bt = rolling_origin_backtest(feats, cols, cfg['params'], cfg['num_boost_round'],
                                         BACKTEST_START, EVAL_END, retrain_every_days=14)
            bt.to_csv(bt_file, index=False)
        results[zone] = evaluate(bt)
        print(json.dumps(results[zone], indent=2), flush=True)

        final = feats[(feats['delivery_date'] <= LAST_KNOWN_DAY) & feats[TARGET].notna()]
        p45 = fit_quantile_models(final[cols], final[TARGET], cfg['params'], cfg['num_boost_round'])['p45']
        gain = pd.Series(p45.feature_importance('gain'), index=cols)
        importances[zone] = (gain / gain.sum()).sort_values(ascending=False).head(10).round(4).to_dict()

        rp = replay_live_day(raw, zone, cfg, bt)
        rp.insert(0, 'zone', zone)
        replays.append(rp)

    # Live day: honest replay vs. the originally submitted (leaky) forecast
    replay = pd.concat(replays)
    sub = pd.read_csv(OUT / 'forecasts' / 'heal_the_grid_predictions.csv')
    sub['timestamp'] = pd.to_datetime(sub['timestamp'], utc=True)
    live = {}
    for zone in ZONES:
        r = replay[replay['zone'] == zone].reset_index(drop=True)
        s = sub.set_index('timestamp').reindex(r['timestamp']).reset_index(drop=True)
        live[zone] = {
            'pinball_q45_model': pinball_loss(r['y'], r['p45']),
            'pinball_q45_original_submission': pinball_loss(r['y'], s[f'{zone} p50']),
            'pinball_q45_naive_d1': pinball_loss(r['y'], r['naive_d1']),
            'pinball_q45_naive_d7': pinball_loss(r['y'], r['naive_d7']),
            'coverage_95_cqr': float(np.mean((r['y'] >= r['p025_cqr']) & (r['y'] <= r['p975_cqr']))),
            'coverage_95_original_submission': float(np.mean((r['y'] >= s[f'{zone} p025']) & (r['y'] <= s[f'{zone} p975']))),
        }
    live['both_zones_avg'] = {k: float(np.mean([live[z][k] for z in ZONES])) for k in live['DE-LU']}
    replay.to_csv(OUT / 'forecasts' / 'day_ahead_replay_2026-05-11.csv', index=False)

    summary = {'evaluation_window': [EVAL_START, EVAL_END], 'backtest': results,
               'live_day_2026-05-11': live, 'top_features_gain_share': importances,
               'quantiles': QUANTILES}
    (OUT / 'backtest' / 'metrics.json').write_text(json.dumps(summary, indent=2))
    print('\nLive day 2026-05-11:', json.dumps(live, indent=2))
    print('\nTop features:', json.dumps(importances, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retune', action='store_true', help='re-run Optuna even if params exist')
    parser.add_argument('--reuse-backtest', action='store_true', help='load saved backtest predictions')
    args = parser.parse_args()
    main(args.retune, args.reuse_backtest)
