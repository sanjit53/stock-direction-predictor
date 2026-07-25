import os    # Checks if file exists
import yfinance as yf  # Download stock market data
import pandas as pd  # Work with tables 
from datetime import date, timedelta

table = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


#Get historical price data for a list of tickers between two dates. 
# Returns a dataframe with the following columns: date, ticker, open, high, low, close, adj_close, volume
def get_price_data(tickers,start,end):
    # Make sure at list one ticker is provided
    if not tickers:
        raise ValueError("tickers must be a non-empty list of ticker strings")

    raw = yf.download(
        tickers,
        start = start,
        end = end,
        group_by = "ticker",
        auto_adjust = False,
    )
    # Return an empty table if not data was found
    if raw.empty:
        print("Warning: no data returned for any ticker in this date range.")
        return pd.DataFrame(columns = table)

    dataframe = []

    for i in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            # Get only this ticker's data
            if i not in raw.columns.get_level_values(0):
                print(f"Warning: no data returned for {i}.")
                continue
            ticker_data = raw[i].copy()
        else:
            ticker_data = raw.copy()
        # Remove empty rows
        ticker_data = ticker_data.dropna(how = "all")

        if ticker_data.empty:
            print(f"Warning: no data return for {i}.")
            continue
        ticker_data = ticker_data.rename(columns ={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
        })

        # Turn index into a normal date column
        ticker_data = ticker_data.reset_index().rename(columns = {"Date" : "date"})
        ticker_data["date"] = pd.to_datetime(ticker_data["date"]).dt.date 
        ticker_data["ticker"] = i
        # Remove rows missing important price data
        ticker_data = ticker_data.dropna(subset=["open", "high", "low", "close", "adj_close"])

        dataframe.append(ticker_data[table])
    # Returns an empty dataframe if no valid data was found
    if not dataframe:
        print("Warning: none of the requested tickers returned usable data.")
        return pd.DataFrame(columns = table)

    result = pd.concat(dataframe, ignore_index = True)
    result = result.drop_duplicates(subset = ["date", "ticker"])
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)

    return result


# Update an existing CSV with new price data
def update_historical_csv(tickers, csv_path):
    today = date.today()
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No file found at {csv_path}."
    )
    
    past_data = pd.read_csv(csv_path, parse_dates=["date"])
    past_data["date"] = past_data["date"].dt.date
    
    new_rows_list = []

    for i in tickers:
        ticker_rows = past_data[past_data["ticker"] == i]
        
        if ticker_rows.empty:
            raise ValueError()
        # Find the most recent saved date 
        last_date = ticker_rows["date"].max()

        if last_date >= today:
            print(f"{i} is up to date (last date: {last_date}).")
            continue
        #Start downloading from the next missing day
        pull_start = last_date + timedelta(days = 1)

        
        new_rows = get_price_data([i], start = pull_start, end = today)
        if not new_rows.empty:
            new_rows_list.append(new_rows)
    # Return the existing data if nothing new was found
    if not new_rows_list:
        print("No new data to add. CSV is current.")
        return past_data[table]
    
    past_data = pd.concat([past_data] + new_rows_list, ignore_index = True)
    past_data = past_data.drop_duplicates(subset=["date", "ticker"])
    past_data = past_data.sort_values(["ticker", "date"]).reset_index(drop = True)

    past_data.to_csv(csv_path, index = False)
 
    return past_data

# Creates engineered features and a binary target for next-day stock price prediction.
def get_features_and_target(df):
    required_cols = {"date", "ticker", "adj_close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required column(s): {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
   
    # Sort data by ticker and date to preserve time order.
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    ticker_frames = []

    # Calculate features independently for each stock to avoid mixing time series.
    for t in df["ticker"].unique():
        group = df[df["ticker"] == t].copy()

        group["ma_5"] = group["adj_close"].rolling(window=5).mean()
        group["ma_20"] = group["adj_close"].rolling(window=20).mean()

        group["momentum_5"] = (group["adj_close"] - group["adj_close"].shift(5)).abs()

        daily_return = group["adj_close"].pct_change()
        group["volatility_10"] = daily_return.rolling(window=10).std()

        group["volume_ma_10"] = group["volume"].rolling(window=10).mean()

        # Target is 1 if tomorrow's closing price is higher than today's.
        next_close = group["adj_close"].shift(-1)
        group["target"] = (next_close > group["adj_close"]).astype(float)
        group.loc[next_close.isna(), "target"] = float("nan")

        ticker_frames.append(group)

    working = pd.concat(ticker_frames, ignore_index=True)
    
    # Remove rows missing rolling-window features or the final target value.
    working = working.dropna()
    working["target"] = working["target"].astype(int)

    # Re-sort by date so the train/test split remains chronological across all stocks.
    working = working.sort_values("date").reset_index(drop=True)

    drop_cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "target"]

    dates = working["date"].reset_index(drop=True)
    target = working["target"]
    features = working.drop(columns=[c for c in drop_cols if c in working.columns])

    return features, target, dates

# Splits the dataset into chronological training and testing sets (80/20).
def get_train_test_split(features, target, dates):
    if len(features) != len(target) or len(features) != len(dates):
        raise ValueError("features, target, and dates must all be the same length")

    unique_dates = pd.Series(dates.unique()).sort_values().reset_index(drop=True)
    cutoff_idx = int(len(unique_dates) * 0.8) - 1
    cutoff_date = unique_dates.iloc[cutoff_idx]

    train_mask = dates <= cutoff_date
    test_mask = dates > cutoff_date

    train_features = features[train_mask]
    train_target = target[train_mask]
    test_features = features[test_mask]
    test_target = target[test_mask]

    train_dates = dates[train_mask]
    test_dates = dates[test_mask]

    # Verify that no future data leaks into the training set.
    assert train_dates.max() < test_dates.min(), (
        "Split is not chronological — test data overlaps with or precedes training data."
    )

    return train_features, train_target, test_features, test_target
