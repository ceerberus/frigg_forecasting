"""
LightGBM quantile regression for day-ahead prices, with a rolling-origin backtest
and conformalized quantile regression (CQR) for the prediction intervals.
"""

from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.day_ahead import TARGET

# p45 is the point forecast: the 0.45-quantile minimises expected pinball loss at q=0.45.
QUANTILES = {'p025': 0.025, 'p45': 0.45, 'p975': 0.975}
ALPHA = 0.05


def fit_quantile_models(X: pd.DataFrame, y: pd.Series, params: dict, num_boost_round: int) -> Dict[str, lgb.Booster]:
    base = {'objective': 'quantile', 'verbosity': -1, 'bagging_freq': 1, 'seed': 42, **params}
    return {name: lgb.train({**base, 'alpha': q}, lgb.Dataset(X, y), num_boost_round=num_boost_round)
            for name, q in QUANTILES.items()}


def predict_quantiles(models: Dict[str, lgb.Booster], X: pd.DataFrame) -> pd.DataFrame:
    """Predict all quantiles; sorting per row removes quantile crossing."""
    raw = np.column_stack([models[name].predict(X) for name in QUANTILES])
    return pd.DataFrame(np.sort(raw, axis=1), columns=list(QUANTILES), index=X.index)


def cqr_scores(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """CQR conformity scores (Romano et al., 2019): signed distance outside the interval."""
    return np.maximum(lower - y, y - upper)


def cqr_margin(scores: np.ndarray, alpha: float = ALPHA) -> float:
    """Finite-sample-corrected (1 - alpha) empirical quantile of the conformity scores."""
    scores = scores[~np.isnan(scores)]
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method='higher'))


def rolling_origin_backtest(
    features: pd.DataFrame,
    feature_cols: List[str],
    params: dict,
    num_boost_round: int,
    start: str,
    end: str,
    retrain_every_days: int = 7,
    calibration_days: int = 90,
    min_calibration_days: int = 30,
) -> pd.DataFrame:
    """
    Expanding-window backtest: for each delivery day D in [start, end], train on all
    days < D (refit every `retrain_every_days`) and forecast the 24 hours of D.

    Intervals are conformalized online: the CQR margin for day D is computed from the
    out-of-sample scores of the previous `calibration_days` days only. Days before
    `min_calibration_days` of history exist get NaN conformal bounds.
    """
    days = pd.date_range(start, end, freq='D')
    labelled = features[features[TARGET].notna()]
    preds, models, last_fit = [], None, None

    for day in days:
        if models is None or (day - last_fit).days >= retrain_every_days:
            train = labelled[labelled['delivery_date'] < day]
            models = fit_quantile_models(train[feature_cols], train[TARGET], params, num_boost_round)
            last_fit = day
        test = features[features['delivery_date'] == day]
        if test.empty:
            continue
        p = predict_quantiles(models, test[feature_cols])
        p['timestamp'] = pd.DatetimeIndex(test['timestamp'])
        p['delivery_date'] = day
        p['y'] = test[TARGET].values
        p['naive_d1'] = test['price_same_hour_ref'].values
        p['naive_d7'] = test['price_same_hour_d7'].values
        preds.append(p)

    out = pd.concat(preds, ignore_index=True)
    out['score'] = cqr_scores(out['y'].values, out['p025'].values, out['p975'].values)

    margins = {}
    for day in days:
        window = out[(out['delivery_date'] < day) & (out['delivery_date'] >= day - pd.Timedelta(days=calibration_days))]
        n_days = window['delivery_date'].nunique()
        margins[day] = cqr_margin(window['score'].values) if n_days >= min_calibration_days else np.nan
    out['cqr_margin'] = out['delivery_date'].map(margins)
    out['p025_cqr'] = out['p025'] - out['cqr_margin']
    out['p975_cqr'] = out['p975'] + out['cqr_margin']
    return out


def tune_params(
    features: pd.DataFrame,
    feature_cols: List[str],
    train_end: str,
    valid_end: str,
    n_trials: int = 40,
    seed: int = 42,
) -> dict:
    """
    Optuna (TPE) search on the primary metric (pinball loss at q=0.45).
    Uses a validation window strictly before the backtest so tuning cannot leak into it.
    Returns the params plus the early-stopped number of boosting rounds.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    labelled = features[features[TARGET].notna()]
    train = labelled[labelled['delivery_date'] < pd.Timestamp(train_end)]
    valid = labelled[(labelled['delivery_date'] >= pd.Timestamp(train_end))
                     & (labelled['delivery_date'] < pd.Timestamp(valid_end))]
    dtrain = lgb.Dataset(train[feature_cols], train[TARGET], params={'feature_pre_filter': False})
    dvalid = lgb.Dataset(valid[feature_cols], valid[TARGET], reference=dtrain)

    def objective(trial):
        params = {
            'objective': 'quantile', 'alpha': QUANTILES['p45'], 'metric': 'quantile',
            'verbosity': -1, 'bagging_freq': 1, 'seed': seed,
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 8, 128, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 300, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10, log=True),
        }
        booster = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dvalid],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        trial.set_user_attr('num_boost_round', booster.best_iteration)
        return booster.best_score['valid_0']['quantile']

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return {'params': study.best_params,
            'num_boost_round': study.best_trial.user_attrs['num_boost_round'],
            'valid_pinball_q45': study.best_value}
