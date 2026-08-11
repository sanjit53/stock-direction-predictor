import os
import sys
import joblib
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))

from functions import update_data              # data/functions.py
from modeling_functions import get_latest_features   # modeling/modeling_functions.py


def make_predictions(ticker="AAPL"):
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "market_data.csv")

    # 1. Make sure the CSV has the most recent price data
    update_data([ticker], csv_path=csv_path)

    # 2. Load the trained model
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    model = joblib.load(model_path)

    # 3. Pull the most recent rows for this ticker from the CSV
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df[df["ticker"] == ticker].copy()

    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}' in {csv_path}")

    # 4. Run them through feature engineering to get the latest feature row
    features, latest_date = get_latest_features(df)

    # 5. Return the model's prediction
    prediction = model.predict(features)[0]

    return {
        "ticker": ticker,
        "date": latest_date,
        "prediction": int(prediction),
    }


if __name__ == "__main__":
    result = make_predictions("AAPL")
    print(result)