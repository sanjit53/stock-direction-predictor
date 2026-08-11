import pandas as pd


def get_features_and_target(df):
    required_cols = {"date", "ticker", "adj_close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required column(s): {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    ticker_frames = []

    for t in df["ticker"].unique():
        group = df[df["ticker"] == t].copy()

        group["ma_5"] = group["adj_close"].rolling(window=5).mean()
        group["ma_20"] = group["adj_close"].rolling(window=20).mean()

        group["momentum_5"] = (group["adj_close"] - group["adj_close"].shift(5)).abs()

        daily_return = group["adj_close"].pct_change()
        group["volatility_10"] = daily_return.rolling(window=10).std()

        group["volume_ma_10"] = group["volume"].rolling(window=10).mean()

        next_close = group["adj_close"].shift(-1)
        group["target"] = (next_close > group["adj_close"]).astype(float)
        group.loc[next_close.isna(), "target"] = float("nan")

        ticker_frames.append(group)

    working = pd.concat(ticker_frames, ignore_index=True)
    working = working.dropna()
    working["target"] = working["target"].astype(int)

    working = working.sort_values("date").reset_index(drop=True)

    drop_cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "target"]

    dates = working["date"].reset_index(drop=True)
    target = working["target"]
    features = working.drop(columns=[c for c in drop_cols if c in working.columns])

    return features, target, dates


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

    assert train_dates.max() < test_dates.min(), (
        "Split is not chronological — test data overlaps with or precedes training data."
    )

    return train_features, train_target, test_features, test_target


# Computes the same rolling features as get_features_and_target, but for a single
# ticker's most recent row — no target column required (there's no "next day" yet
# for a live prediction), so rows aren't dropped for missing target.
def get_latest_features(df):
    required_cols = {"date", "ticker", "adj_close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required column(s): {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["ma_5"] = df["adj_close"].rolling(window=5).mean()
    df["ma_20"] = df["adj_close"].rolling(window=20).mean()
    df["momentum_5"] = (df["adj_close"] - df["adj_close"].shift(5)).abs()

    daily_return = df["adj_close"].pct_change()
    df["volatility_10"] = daily_return.rolling(window=10).std()

    df["volume_ma_10"] = df["volume"].rolling(window=10).mean()

    # Drop only rows where the rolling features aren't fully formed yet
    feature_cols = ["ma_5", "ma_20", "momentum_5", "volatility_10", "volume_ma_10"]
    df = df.dropna(subset=feature_cols)

    if df.empty:
        raise ValueError("Not enough historical data to compute features for this ticker.")

    latest_row = df.iloc[[-1]]
    latest_date = latest_row["date"].values[0]

    drop_cols = ["date", "ticker", "open", "high", "low", "close", "adj_close"]
    features = latest_row.drop(columns=[c for c in drop_cols if c in latest_row.columns])

    return features, latest_date