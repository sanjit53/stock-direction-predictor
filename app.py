import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make data/ and modeling/ importable no matter where the app is launched from
ROOT = Path(__file__).parent
sys.path.append(str(ROOT / "data"))
sys.path.append(str(ROOT / "modeling"))

from predictions import make_predictions            # modeling/predictions.py
from modeling_functions import get_latest_features  # modeling/modeling_functions.py

CSV_PATH = ROOT / "data" / "market_data.csv"


def load_data():
    # Not cached on purpose: make_predictions() appends new rows to the CSV,
    # so re-reading each run keeps the chart and metric current.
    return pd.read_csv(CSV_PATH, parse_dates=["date"])


st.title("Stock Direction Predictor")

# ---------- Step 1: ticker input ----------
ticker = st.text_input("Ticker symbol", "AAPL").strip().upper()

df = load_data()
ticker_df = df[df["ticker"] == ticker].sort_values("date")

if ticker_df.empty:
    available = ", ".join(sorted(df["ticker"].unique()))
    st.warning(f"No data for {ticker}. Tickers available: {available}")
    st.stop()

# ---------- Step 2: latest price metric ----------
latest = ticker_df.iloc[-1]
previous = ticker_df.iloc[-2]
st.metric(
    label=f"{ticker} adj close ({latest['date']:%b %d, %Y})",
    value=f"${latest['adj_close']:.2f}",
    delta=f"{latest['adj_close'] - previous['adj_close']:.2f}",
)

# ---------- Step 1: last 90 days of adj_close, read from the CSV ----------
cutoff = ticker_df["date"].max() - pd.Timedelta(days=90)
last_90 = ticker_df[ticker_df["date"] >= cutoff]
st.line_chart(last_90.set_index("date")["adj_close"])

# ---------- Step 1: prediction button ----------
if st.button("Run Prediction"):
    try:
        with st.spinner("Updating data and running the model..."):
            result = make_predictions(ticker)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    if result["prediction"] == 1:
        st.success("Prediction: UP ↑")
    else:
        st.error("Prediction: DOWN ↓")
    st.caption(
        f"Model: {result['model']} · probability of UP: {result['probability_up']:.0%} "
        f"· based on data through {pd.Timestamp(result['date']):%b %d, %Y}"
    )

    # ---------- Step 2: the exact feature row the model used ----------
    st.subheader("Model Inputs")
    fresh = load_data()  # re-read: make_predictions may have added new rows
    features, latest_date = get_latest_features(fresh[fresh["ticker"] == ticker])
    features.insert(0, "date", pd.Timestamp(latest_date).date())
    st.dataframe(features, hide_index=True)