import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from functions import get_price_data, get_features_and_target, get_train_test_split


def load_data(tickers, start, end, csv_path=None):
    """
    Load price data either from an existing CSV (fast, no network calls)
    or fresh from Yahoo Finance via get_price_data.
    """
    if csv_path:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df[df["ticker"].isin(tickers)].copy()
    else:
        df = get_price_data(tickers, start=start, end=end)
    return df


def train_logistic_regression(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    return model, preds


def train_random_forest(X_train, y_train, X_test, y_test, random_state=42):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return model, preds


def evaluate(name, y_test, preds):
    print(f"\n=== {name} ===")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, preds, zero_division=0):.4f}")
    print(f"F1 score:  {f1_score(y_test, preds, zero_division=0):.4f}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))


def main():
    tickers = ["AAPL", "MSFT", "NVDA"]
    start = "2018-01-01"
    end = "2026-01-01"

    df = load_data(tickers, start, end, csv_path="market_data.csv")

    features, target, dates = get_features_and_target(df)
    X_train, y_train, X_test, y_test = get_train_test_split(features, target, dates)

    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
    print(f"Feature columns: {list(features.columns)}")

    log_reg, log_reg_preds = train_logistic_regression(X_train, y_train, X_test, y_test)
    evaluate("Logistic Regression", y_test, log_reg_preds)

    rf, rf_preds = train_random_forest(X_train, y_train, X_test, y_test)
    evaluate("Random Forest", y_test, rf_preds)

    importances = pd.Series(rf.feature_importances_, index=features.columns)
    print("\nRandom Forest feature importances:")
    print(importances.sort_values(ascending=False))


if __name__ == "__main__":
    main()