import pandas as pd
from datetime import timedelta
from sklearn.neighbors import KNeighborsClassifier

from ..storage import get_cache_store
from ..market_data import MarketDataProvider, YFinanceProvider

_default_provider = YFinanceProvider()
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from lightgbm import LGBMClassifier


def get_ticker(ticker):
    return _default_provider.get_raw_ticker(ticker)


def _index_cache_key(ticker, period, start_date=None):
    if start_date is not None:
        return f"index:{ticker}:{period}:{start_date}"
    return f"index:{ticker}:{period}"


def _index_data_fresh(df):
    if df.empty:
        return False
    try:
        latest_date = pd.Timestamp(df.index.max())
        days_old = (pd.Timestamp.now() - latest_date).days
        return -1 <= days_old <= 1
    except (ValueError, TypeError, AttributeError):
        return False


def get_index_data(ticker, period='10y', start_date=None, provider: MarketDataProvider | None = None):
    """Get index data with proper cache handling."""
    _provider = provider if provider is not None else _default_provider
    store = get_cache_store()
    parsed_start = pd.Timestamp(start_date).date() if start_date is not None else None
    key = _index_cache_key(ticker, period, parsed_start)
    max_age = timedelta(hours=12)

    print(f"Attempting to get data for {ticker} over {period}")

    cached = store.get(key, max_age=max_age, validator=_index_data_fresh)
    if cached is not None:
        latest_date = pd.Timestamp(cached.index.max()).date()
        print(f"Using cached data (latest date: {latest_date})")
        return cached

    try:
        lookback = int(period.split('y')[0])

        if parsed_start is not None:
            print(f"Using start date: {parsed_start}")
            end_date = parsed_start
            download_start = pd.Timestamp(end_date) - pd.DateOffset(years=lookback)
            index_data = _provider.get_history(
                ticker,
                start=str(download_start.date()),
                end=str(end_date),
            )
        else:
            print(f"Attempting direct download with period={period}")
            index_data = _provider.get_history(ticker, period=period)

            if index_data.empty:
                print("Period download failed, trying with explicit dates")
                end_date = pd.Timestamp.now()
                download_start = end_date - pd.Timedelta(days=365 * lookback)
                print(f"Downloading from {download_start} to {end_date}")
                index_data = _provider.get_history(
                    ticker,
                    start=str(download_start.date()),
                    end=str(end_date.date()),
                )

            if index_data.empty:
                print("Explicit dates failed, trying max period")
                index_data = _provider.get_history(ticker, period='max')

        if index_data.empty:
            raise ValueError(f"No data retrieved for {ticker}")

        index_data.columns = index_data.columns.get_level_values(0)
        print(f"Saving data to cache: {key}")
        store.put(key, index_data)
        return index_data

    except Exception as e:
        print(f"Error retrieving {ticker} data: {e}")
        try:
            print("Attempting final fallback download")
            return _provider.get_history(ticker, period=period)
        except Exception as download_error:
            print(f"Error downloading {ticker} data: {download_error}")
            return None


def get_indexes_data(indexes, period='10y'):
    """Fetch data for multiple indexes/tickers and return as a list.

    Args:
        indexes: List of index/ticker symbols
        period: Time period to fetch data for (default '10y')

    Returns:
        List of pandas DataFrames containing the data for each index
    """
    data_list = []
    for index in indexes:
        data = get_index_data(index, period)
        if data is not None:
            data_list.append(data)
    return data_list


def get_ticker_data(ticker, provider: MarketDataProvider | None = None):
    _provider = provider if provider is not None else _default_provider
    return _provider.get_history(ticker, period="20y")


def get_ticker_financials(ticker: str, provider: MarketDataProvider | None = None) -> pd.DataFrame:
    _provider = provider if provider is not None else _default_provider
    return _provider.get_financials(ticker)


def get_ticker_quarterly_financials(ticker: str) -> pd.DataFrame:
    return _default_provider.get_quarterly_financials(ticker)


def get_quarterly_balance_sheet(ticker: str, provider=None) -> pd.DataFrame:
    _provider = provider if provider is not None else _default_provider
    return _provider.get_quarterly_balance_sheet(ticker)


def get_significant_changes(data, filter_consecutive=False, filter_return=0.03):
    """Get significant price changes from the data."""
    # Calculate returns first
    data = data.copy()
    if 'return' not in data.columns:
        data['return'] = data['Close'].pct_change(fill_method=None)

    significant_changes = data[abs(data['return']) > filter_return].copy()
    significant_changes['prior_day_return'] = data['return'].shift(1)
    significant_changes['prior_5day_return'] = data['return'].shift(5)
    significant_changes['prior_month_return'] = data['Close'].pct_change(
        periods=21, fill_method=None)
    significant_changes['RSI_5d'] = RSI(data, window=5)

    # Calculate time since last significant change
    all_dates = pd.Series(data.index)
    significant_dates = pd.Series(significant_changes.index)

    # For each significant date, find the number of days since the previous significant date
    time_deltas = []
    for date in significant_changes.index:
        # Find the most recent previous significant date
        prev_dates = significant_dates[significant_dates < date]
        if len(prev_dates) > 0:
            last_sig_date = prev_dates.iloc[-1]
            delta = (date - last_sig_date).days
        else:
            delta = None  # or 0, depending on your preference
        time_deltas.append(delta)

    significant_changes['time_since_last_sig_change'] = time_deltas

    if filter_consecutive:
        significant_changes['prev_day_significant'] = significant_changes.index.isin(
            significant_changes.index - pd.Timedelta(days=1))
        filtered_points = significant_changes[significant_changes['prev_day_significant']].copy(
        )
        significant_changes = significant_changes[~significant_changes['prev_day_significant']]
        significant_changes = significant_changes.drop(
            'prev_day_significant', axis=1)
        return significant_changes, filtered_points

    return significant_changes



def RSI(data, window=14, adjust=False):
    """Calculate the Relative Strength Index (RSI) for a given stock."""
    data = data.copy()['Close']
    delta = data.diff(1).dropna()
    loss = delta.copy()
    gains = delta.copy()

    gains[gains < 0] = 0
    loss[loss > 0] = 0

    gain_ewm = gains.ewm(com=window - 1, adjust=adjust).mean()
    loss_ewm = abs(loss.ewm(com=window - 1, adjust=adjust).mean())

    RS = gain_ewm / loss_ewm
    RSI = 100 - 100 / (1 + RS)

    return RSI


classifiers = {
    "Nearest Neighbors": KNeighborsClassifier(3),
    "Linear SVM": SVC(kernel="linear", C=0.025),
    "RBF SVM": SVC(gamma=2, C=1),
    "Gaussian Process": GaussianProcessClassifier(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Neural Net": MLPClassifier(alpha=1, max_iter=1000),
    "AdaBoost": AdaBoostClassifier(),
    "Naive Bayes": GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
}

# Define feature sets at the top of the file
BASE_FEATURES = ['prev_day_return', 'prev_2day_return', 'prev_week_return']
EXTRA_FEATURES = ['volatility_5d', 'volatility_21d', 'ma_5d', 'ma_21d',
                  'volume_1d', 'volume_5d', 'rsi_5d', 'rsi_21d']


def get_features(use_extra_features=False):
    """Get the appropriate feature list based on whether extra features are used."""
    if use_extra_features:
        return BASE_FEATURES + EXTRA_FEATURES
    return BASE_FEATURES
