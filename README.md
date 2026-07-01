## Overview
This project downloads historical stock market data from Yahoo Finance using the `yfinance` library. It retrieves price data for one or more stock tickers, cleans and formats the data, and updates an existing CSV file with the latest available stock prices.

## Features
- Download historical stock data for one or more tickers.
- Clean and standardize the data into a consistent format.
- Remove duplicate rows.
- Sort data by ticker and date.
- Update an existing CSV with only the missing dates.


## Files

- `main.py` – Contains the functions for downloading and updating stock data.
- `market_data.csv` – Stores the historical stock price data.

## Functions

### `get_price_data(tickers, start, end)`

Downloads historical stock data for the specified stock tickers.

**Parameters**
- `tickers`: List of stock ticker symbols (example: `["AAPL", "MSFT", "NVDA"]`)
- `start`: Start date
- `end`: End date

**Returns**
A pandas DataFrame containing:

- date
- ticker
- open
- high
- low
- close
- adj_close
- volume

---

### `update_historical_csv(tickers, csv_path)`

Updates an existing CSV file with the latest available stock prices.

**Parameters**
- `tickers`: List of stock ticker symbols.
- `csv_path`: Path to the CSV file.

**Returns**
An updated pandas DataFrame and saves the updated data to the CSV file.
