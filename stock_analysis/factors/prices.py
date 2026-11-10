"""Price-derived inputs to the factor scores."""

import numpy as np
import pandas as pd

MOMENTUM_LOOKBACK = 252  # sessions in twelve months
MOMENTUM_SKIP = 21  # sessions in the most recent month, which is left out


def momentum_12_1(
    close: pd.Series, *, lookback: int = MOMENTUM_LOOKBACK, skip: int = MOMENTUM_SKIP
) -> float:
    """Return from twelve months ago to one month ago, as of the last bar of `close`.

    The latest month is skipped because short-term returns tend to reverse, which
    works against the trend the factor is meant to capture. yfinance `history()`
    closes are dividend-adjusted, so this is a total return. NaN with fewer than
    `lookback + 1` bars.
    """
    close = close.dropna()
    if len(close) <= lookback:
        return float("nan")
    start = float(close.iloc[-1 - lookback])
    end = float(close.iloc[-1 - skip])
    if start <= 0:
        return float("nan")
    return end / start - 1


def split_factor(df: pd.DataFrame, after: pd.Timestamp, through: pd.Timestamp) -> float:
    """Product of the split ratios that took effect after `after`, up to and including `through`.

    Used to bring a share count reported before a split onto the share basis of a
    later date. 1.0 when the frame has no `Stock Splits` column.
    """
    if "Stock Splits" not in df.columns:
        return 1.0
    splits = df["Stock Splits"]
    splits = splits[(splits.index > after) & (splits.index <= through)]
    splits = splits[splits > 0]
    return float(splits.prod()) if not splits.empty else 1.0


def unadjusted_close(df: pd.DataFrame) -> pd.Series:
    """The close as it actually traded, recovered from yfinance's adjusted `Close`.

    Adjusted prices are right for returns but wrong for valuation: a historical
    close is scaled down by every dividend and split that came later, so
    `shares reported then x adjusted close` understates the market value of the
    time, by more the further back and the higher the dividend. This undoes both
    adjustments using the `Dividends` and `Stock Splits` columns that
    `history()` returns. Without those columns `Close` is returned as it is.

    The result for a date depends only on what the price was that day, even
    though later corporate actions are needed to get back to it.
    """
    close = df["Close"].astype(float)
    values = close.to_numpy(copy=True)
    n = len(values)

    if "Dividends" in df.columns and n:
        dividends = df["Dividends"].fillna(0.0).to_numpy(dtype=float)
        # yfinance multiplies every close before an ex-date by
        # (1 - dividend / previous close). Walking backwards, the close just
        # before each ex-date is therefore adjusted / factor-so-far + dividend.
        factor = np.ones(n)
        current = 1.0
        upper = n
        for k in np.flatnonzero(dividends > 0)[::-1]:
            factor[k:upper] = current
            upper = k
            if k == 0 or np.isnan(values[k - 1]) or values[k - 1] <= 0:
                continue
            before = values[k - 1] / current + dividends[k]
            current = values[k - 1] / before
        factor[:upper] = current
        values = values / factor

    if "Stock Splits" in df.columns and n:
        splits = df["Stock Splits"].fillna(0.0).to_numpy(dtype=float)
        ratios = np.where(splits > 0, splits, 1.0)
        # Splits strictly after each bar: the split-day close is already post-split
        later = np.append(np.cumprod(ratios[::-1])[::-1][1:], 1.0)
        values = values * later

    return pd.Series(values, index=close.index, name="Close")
