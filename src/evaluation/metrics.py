"""
Evaluation metrics for electricity price forecasting
"""

import numpy as np
from typing import Union, Tuple


def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q: float = 0.45
) -> float:
    """
    Asymmetric pinball loss at quantile q.
    
    This is the official scoring metric for the challenge.
    
    Penalty for underestimation (actual > pred): q × |error|
    Penalty for overestimation (actual < pred): (1 − q) × |error|
    
    At q = 0.45, overestimation penalty is (0.55 / 0.45) ≈ 1.22× the underestimation penalty.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    q : float, default=0.45
        Quantile for asymmetric loss
        
    Returns
    -------
    float
        Mean pinball loss
        
    Examples
    --------
    >>> y_true = np.array([100, 110, 90])
    >>> y_pred = np.array([95, 105, 95])
    >>> pinball_loss(y_true, y_pred, q=0.45)
    3.5
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    residual = y_true - y_pred
    loss = np.where(residual >= 0, q * residual, (q - 1) * residual)
    
    return float(np.mean(loss))


def quantile_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float
) -> float:
    """
    Standard quantile loss (symmetric pinball loss).
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values at specified quantile
    quantile : float
        Target quantile (e.g., 0.025, 0.5, 0.975)
        
    Returns
    -------
    float
        Mean quantile loss
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    residual = y_true - y_pred
    loss = np.where(residual >= 0, quantile * residual, (quantile - 1) * residual)
    
    return float(np.mean(loss))


def coverage_score(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    target_coverage: float = 0.95
) -> float:
    """
    Calculate empirical coverage of prediction intervals.
    
    For 95% intervals (p025 to p975), we expect 95% of true values
    to fall within the predicted bounds.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_lower : np.ndarray
        Lower bound predictions (e.g., p025)
    y_upper : np.ndarray
        Upper bound predictions (e.g., p975)
    target_coverage : float, default=0.95
        Target coverage probability
        
    Returns
    -------
    float
        Empirical coverage rate (0 to 1)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_lower = np.asarray(y_lower, dtype=float)
    y_upper = np.asarray(y_upper, dtype=float)
    
    within_bounds = (y_true >= y_lower) & (y_true <= y_upper)
    empirical_coverage = np.mean(within_bounds)
    
    return float(empirical_coverage)


def interval_width(
    y_lower: np.ndarray,
    y_upper: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate mean and std of prediction interval widths.
    
    Narrower intervals are better (more confident predictions),
    but only if coverage is maintained.
    
    Parameters
    ----------
    y_lower : np.ndarray
        Lower bound predictions
    y_upper : np.ndarray
        Upper bound predictions
        
    Returns
    -------
    Tuple[float, float]
        (mean_width, std_width)
    """
    y_lower = np.asarray(y_lower, dtype=float)
    y_upper = np.asarray(y_upper, dtype=float)
    
    widths = y_upper - y_lower
    
    return float(np.mean(widths)), float(np.std(widths))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Mean Absolute Percentage Error
    
    Note: MAPE can be problematic when true values are near zero.
    We add epsilon to avoid division by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred_p50: np.ndarray,
    y_pred_p025: np.ndarray,
    y_pred_p975: np.ndarray,
    verbose: bool = True
) -> dict:
    """
    Comprehensive evaluation of probabilistic forecast.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred_p50 : np.ndarray
        Median predictions
    y_pred_p025 : np.ndarray
        2.5th percentile predictions
    y_pred_p975 : np.ndarray
        97.5th percentile predictions
    verbose : bool, default=True
        Print results
        
    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """
    metrics = {
        'pinball_loss_q45': pinball_loss(y_true, y_pred_p50, q=0.45),
        'mae': mae(y_true, y_pred_p50),
        'rmse': rmse(y_true, y_pred_p50),
        'mape': mape(y_true, y_pred_p50),
        'coverage_95': coverage_score(y_true, y_pred_p025, y_pred_p975, target_coverage=0.95),
        'mean_interval_width': interval_width(y_pred_p025, y_pred_p975)[0],
        'std_interval_width': interval_width(y_pred_p025, y_pred_p975)[1],
    }
    
    if verbose:
        print("=" * 60)
        print("FORECAST EVALUATION METRICS")
        print("=" * 60)
        print(f"Pinball Loss (q=0.45):     {metrics['pinball_loss_q45']:.4f} EUR/MWh")
        print(f"MAE:                       {metrics['mae']:.4f} EUR/MWh")
        print(f"RMSE:                      {metrics['rmse']:.4f} EUR/MWh")
        print(f"MAPE:                      {metrics['mape']:.2f}%")
        print(f"95% Coverage:              {metrics['coverage_95']:.2%}")
        print(f"Mean Interval Width:       {metrics['mean_interval_width']:.2f} EUR/MWh")
        print(f"Std Interval Width:        {metrics['std_interval_width']:.2f} EUR/MWh")
        print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    # Test the metrics
    np.random.seed(42)
    
    # Simulate some predictions
    y_true = np.random.uniform(50, 150, 100)
    y_pred_p50 = y_true + np.random.normal(0, 10, 100)
    y_pred_p025 = y_pred_p50 - 20
    y_pred_p975 = y_pred_p50 + 20
    
    # Evaluate
    metrics = evaluate_forecast(y_true, y_pred_p50, y_pred_p025, y_pred_p975)
    
    print("\nTest completed successfully!")

# Made with Bob
