import yfinance as yf
import pandas as pd
import os
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def get_ticker(ticker):
    return yf.Ticker(ticker)


def get_index_data(ticker, period='10y', start_date=None):
    """Get index data with proper cache handling."""
    CACHE_DIR = f'data/{ticker}_cache'
    cache_file = os.path.join(CACHE_DIR, f'{ticker}_{period}.csv')

    if start_date is not None:
        start_date = pd.Timestamp(start_date).date()
        cache_file = os.path.join(
            CACHE_DIR, f'{ticker}_{period}_{start_date}.csv')

    # Add debug prints
    print(f"Attempting to get data for {ticker} over {period}")

    try:
        lookback = int(period.split('y')[0])
        if start_date is not None:
            print(f"Using start date: {start_date}")
            if os.path.exists(cache_file):
                file_age_ok = os.path.getmtime(cache_file) > pd.Timestamp.now().timestamp() - (12 * 60 * 60)
                
                if file_age_ok:
                    try:
                        cached_data = pd.read_csv(
                            cache_file, index_col=0, parse_dates=True)
                        
                        # Check data freshness
                        if not cached_data.empty:
                            try:
                                latest_date = pd.Timestamp(cached_data.index.max())
                                now = pd.Timestamp.now()
                                days_old = (now - latest_date).days
                                
                                if days_old > 1:
                                    print(f"Cache data is {days_old} days old (latest date: {latest_date.date()}), invalidating cache")
                                elif days_old < -1:
                                    print(f"Cache data has future dates (latest: {latest_date.date()}), invalidating cache")
                                else:
                                    print(f"Using cached data (latest date: {latest_date.date()})")
                                    return cached_data
                            except (ValueError, TypeError, AttributeError) as e:
                                print(f"Error checking cache data dates: {e}, invalidating cache")
                        else:
                            print("Cached data is empty, invalidating cache")
                    except Exception as e:
                        print(f"Error reading cache file: {e}, invalidating cache")
                else:
                    print("Cache file is outdated (file modification time)")
            else:
                print("Cache file does not exist or is outdated")
                end_date = start_date
                start_date = end_date - pd.DateOffset(years=lookback)
                index_data = yf.download(ticker, start=start_date, end=end_date)
                os.makedirs(CACHE_DIR, exist_ok=True)
                index_data.columns = index_data.columns.get_level_values(
                    0)  # Remove multi-index if present
                index_data.to_csv(cache_file)
                return index_data

        # Check cache first
        if os.path.exists(cache_file):
            print(f"Found cache file: {cache_file}")
            # Check both file modification time and data freshness
            file_age_ok = os.path.getmtime(cache_file) > pd.Timestamp.now().timestamp() - (12 * 60 * 60)
            
            if file_age_ok:
                # Read the cached data to check data freshness
                try:
                    cached_data = pd.read_csv(
                        cache_file, index_col=0, parse_dates=True)
                    
                    # Check if data is recent enough (latest date should be within 1 day of today)
                    if not cached_data.empty:
                        try:
                            latest_date = pd.Timestamp(cached_data.index.max())
                            # Check if it's a valid date (not too far in the future or past)
                            now = pd.Timestamp.now()
                            days_old = (now - latest_date).days
                            
                            # If data is more than 1 day old, invalidate cache
                            if days_old > 1:
                                print(f"Cache data is {days_old} days old (latest date: {latest_date.date()}), invalidating cache")
                            elif days_old < -1:
                                print(f"Cache data has future dates (latest: {latest_date.date()}), invalidating cache")
                            else:
                                print(f"Using cached data (latest date: {latest_date.date()})")
                                return cached_data
                        except (ValueError, TypeError, AttributeError) as e:
                            print(f"Error checking cache data dates: {e}, invalidating cache")
                    else:
                        print("Cached data is empty, invalidating cache")
                except Exception as e:
                    print(f"Error reading cache file: {e}, invalidating cache")
            else:
                print("Cache file is outdated (file modification time)")
                print(time.ctime(os.path.getmtime(cache_file)))

        # Cache doesn't exist or is outdated, try downloading with period
        print(f"Attempting direct download with period={period}")
        index_data = yf.download(ticker, period=period)

        if index_data.empty:
            print("Period download failed, trying with explicit dates")
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=365 * lookback)
            print(f"Downloading from {start_date} to {end_date}")
            index_data = yf.download(ticker, start=start_date, end=end_date)

        if index_data.empty:
            print("Explicit dates failed, trying max period")
            index_data = yf.download(ticker, period='max')

        if index_data.empty:
            raise ValueError(f"No data retrieved for {ticker}")

        # Save to cache - ensure data is in the correct format
        print(f"Saving data to cache: {cache_file}")
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Clean up column names and save
        index_data.columns = index_data.columns.get_level_values(
            0)  # Remove multi-index if present
        index_data.to_csv(cache_file)

        return index_data

    except Exception as e:
        print(f"Error retrieving {ticker} data: {e}")
        # If all else fails, try one last direct download
        try:
            print("Attempting final fallback download")
            return yf.download(ticker, period=period)
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


def get_ticker_data(ticker):
    return get_ticker(ticker).history(period="20y")


def get_ticker_financials(ticker):
    return get_ticker(ticker).financials


def get_significant_changes(data, filter_consecutive=False, filter_return=0.03):
    """Get significant price changes from the data."""
    # Calculate returns first
    data = data.copy()
    if 'return' not in data.columns:
        data['return'] = data['Close'].pct_change()

    significant_changes = data[abs(data['return']) > filter_return].copy()
    significant_changes['prior_day_return'] = data['return'].shift(1)
    significant_changes['prior_5day_return'] = data['return'].shift(5)
    significant_changes['prior_month_return'] = data['Close'].pct_change(
        periods=21)
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
    "QDA": QuadraticDiscriminantAnalysis()
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
