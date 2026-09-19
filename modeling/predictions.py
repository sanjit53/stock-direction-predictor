import os
import sys

import joblib
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))

from functions import update_historical_csv          # data/functions.py
from modeling_functions import get_latest_features    # modeling/modeling_functions.py

MODEL_FILES = {
    "random_forrest": "random_forrest.pkl",
    "logistic_regression": "logistic_regression.pkl",
}


def make_predictions(ticker="AAPL", model="random_forrest"):
    if model not in MODEL_FILES:
        raise ValueError(f"model must be one of {list(MODEL_FILES)}, got '{model}'")

    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "market_data.csv")

    # 1. Make sure the CSV has the most recent price data
    update_historical_csv([ticker], csv_path=csv_path)

    # 2. Load the requested trained model (bundled with its scaler, if any,
    #    and the exact feature column order it was trained on)
    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILES[model])
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model found at {model_path}. Run training.py first."
        )
    saved = joblib.load(model_path)
    trained_model = saved["model"]
    scaler = saved.get("scaler")
    feature_columns = saved.get("feature_columns")

    # 3. Pull the most recent rows for this ticker from the CSV
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df[df["ticker"] == ticker].copy()

    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}' in {csv_path}")

    # 4. Run them through feature engineering to get the latest feature row
    features, latest_date = get_latest_features(df)

    if feature_columns is not None:
        features = features.reindex(columns=feature_columns)

    X = features
    if scaler is not None:
        X = scaler.transform(X)

    # 5. Return the model's prediction — same output shape for either model
    prediction = int(trained_model.predict(X)[0])
    probability_up = float(trained_model.predict_proba(X)[0][1])

    return {
        "ticker": ticker,
        "model": model,
        "date": latest_date,
        "prediction": prediction,
        "prediction_label": "up" if prediction == 1 else "down",
        "probability_up": round(probability_up, 4),
    }


if __name__ == "__main__":
    result = make_predictions("AAPL", "random_forrest")
    print(result)