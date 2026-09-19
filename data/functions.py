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