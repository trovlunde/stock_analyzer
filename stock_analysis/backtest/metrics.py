"""Performance statistics for backtest equity curves, plus overfitting guards."""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from stock_analysis.backtest.engine import BacktestResult

TRADING_DAYS_PER_YEAR = 252
_EULER_MASCHERONI = 0.5772156649015329


def daily_sharpe(returns: pd.Series) -> float | None:
    """Per-period (not annualised) Sharpe ratio; None when returns never vary."""
    if len(returns) < 2:
        return None
    std = returns.std(ddof=1)
    if not std or np.isnan(std):
        return None
    return float(returns.mean() / std)


def performance_metrics(equity: pd.Series) -> dict:
    """Return and risk statistics for a daily equity curve (risk-free rate of zero)."""
    returns = equity.pct_change().dropna()
    years = len(returns) / TRADING_DAYS_PER_YEAR
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else None

    sharpe_daily = daily_sharpe(returns)
    downside = math.sqrt(float((returns.clip(upper=0) ** 2).mean())) if len(returns) else 0.0
    max_drawdown = float((equity / equity.cummax() - 1).min())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
        if len(returns) > 1 else None,
        "sharpe": sharpe_daily * math.sqrt(TRADING_DAYS_PER_YEAR)
        if sharpe_daily is not None else None,
        "sortino": float(returns.mean() / downside * math.sqrt(TRADING_DAYS_PER_YEAR))
        if downside > 0 else None,
        "max_drawdown": max_drawdown,
        "calmar": cagr / abs(max_drawdown) if cagr is not None and max_drawdown < 0 else None,
    }


def trade_statistics(result: BacktestResult) -> dict:
    """Trading activity and cost statistics for a backtest."""
    years = (len(result.equity) - 1) / TRADING_DAYS_PER_YEAR
    traded_notional = sum(trade.notional for trade in result.trades)
    total_costs = sum(trade.costs for trade in result.trades)
    closed = [t.realized_pnl for t in result.trades if t.realized_pnl is not None]

    return {
        "trades": len(result.trades),
        "round_trips": len(closed),
        "hit_rate": sum(pnl > 0 for pnl in closed) / len(closed) if closed else None,
        # Traded notional per year as a multiple of average portfolio value
        "annual_turnover": traded_notional / float(result.equity.mean()) / years
        if years > 0 else None,
        "total_costs": total_costs,
        "cost_drag": total_costs / result.initial_cash,
        "avg_exposure": float(result.exposure.mean()),
    }


def paired_block_bootstrap(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    block_size: int = 21,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict:
    """Bootstrap the strategy's Sharpe ratio and how often it beats the benchmark's.

    Resamples blocks of consecutive days (keeping short-term autocorrelation) and
    uses the same days for both series, so the comparison is paired.
    """
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    values = aligned.to_numpy()
    n = len(values)
    if n < block_size * 2:
        return {"sharpe_ci_low": None, "sharpe_ci_high": None, "prob_beats_benchmark": None}

    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block_size)
    annualise = math.sqrt(TRADING_DAYS_PER_YEAR)
    strategy_sharpes = []
    wins = 0
    comparisons = 0
    for _ in range(n_resamples):
        starts = rng.integers(0, n, size=n_blocks)
        # Circular blocks: a block that runs off the end wraps to the start
        index = ((starts[:, None] + np.arange(block_size)[None, :]) % n).ravel()[:n]
        sample = values[index]
        std = sample.std(axis=0, ddof=1)
        if std[0] > 0:
            strategy_sharpe = sample[:, 0].mean() / std[0] * annualise
            strategy_sharpes.append(strategy_sharpe)
            if std[1] > 0:
                comparisons += 1
                wins += strategy_sharpe > sample[:, 1].mean() / std[1] * annualise

    if not strategy_sharpes:
        return {"sharpe_ci_low": None, "sharpe_ci_high": None, "prob_beats_benchmark": None}
    low, high = np.percentile(strategy_sharpes, [5, 95])
    return {
        "sharpe_ci_low": float(low),
        "sharpe_ci_high": float(high),
        "prob_beats_benchmark": wins / comparisons if comparisons else None,
    }


def deflated_sharpe_ratio(returns: pd.Series, trial_sharpes: list[float]) -> float | None:
    """Probability that the Sharpe ratio is positive after allowing for selection.

    `trial_sharpes` are the per-period Sharpe ratios of every configuration tried
    on this data, including this one. The best of many random strategies has a
    high Sharpe by chance; this compares against that expected maximum
    (Bailey & López de Prado, 2014). None when there are too few trials to tell.
    """
    sharpe = daily_sharpe(returns)
    n_trials = len(trial_sharpes)
    if sharpe is None or n_trials < 2:
        return None
    trial_std = float(np.std(trial_sharpes, ddof=1))
    if trial_std == 0:
        return None

    normal = NormalDist()
    expected_max_sharpe = trial_std * (
        (1 - _EULER_MASCHERONI) * normal.inv_cdf(1 - 1 / n_trials)
        + _EULER_MASCHERONI * normal.inv_cdf(1 - 1 / (n_trials * math.e))
    )
    skew = float(returns.skew())
    kurtosis = float(returns.kurt()) + 3  # pandas reports excess kurtosis
    variance_term = 1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2
    if variance_term <= 0:
        return None
    z = (sharpe - expected_max_sharpe) * math.sqrt(len(returns) - 1) / math.sqrt(variance_term)
    return normal.cdf(z)
