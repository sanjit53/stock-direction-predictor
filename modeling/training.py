"""
Train and compare a Logistic Regression model and a Random Forest classifier
to predict next-day stock price direction (up/down), using the feature
pipeline in modeling_functions.py.
"""

import os

import joblib
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

from modeling_functions import get_features_and_target, get_train_test_split

# Maps the "model" argument in make_predictions() to the file it loads.
MODEL_FILES = {
    "random_forrest": "random_forrest.pkl",
    "logistic_regression": "logistic_regression.pkl",
}

# This script's own folder — pickle files are always saved/loaded here,
# regardless of which directory you launch Python from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def train_logistic_regression(X_train, y_train, X_test, y_test):
    # Logistic regression is sensitive to feature scale, so standardize first.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    model.scaler_ = scaler  # stash for train_and_save_models to pick up

    preds = model.predict(X_test_scaled)
    return model, preds


def train_random_forest(X_train, y_train, X_test, y_test, random_state=42):
    # Tree-based models don't need feature scaling.
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


def train_and_save_models(df, model_dir=None):
    """
    Train both models, print their evaluation metrics, and save each as a
    joblib file that predictions.py can load later:
      - logistic_regression.pkl  ({"model", "scaler", "feature_columns"})
      - random_forrest.pkl       ({"model", "scaler": None, "feature_columns"})

    model_dir defaults to this script's own folder.
    """
    if model_dir is None:
        model_dir = SCRIPT_DIR

    features, target, dates = get_features_and_target(df)
    X_train, y_train, X_test, y_test = get_train_test_split(features, target, dates)
    feature_columns = list(features.columns)

    log_reg, log_reg_preds = train_logistic_regression(X_train, y_train, X_test, y_test)
    evaluate("Logistic Regression", y_test, log_reg_preds)

    rf, rf_preds = train_random_forest(X_train, y_train, X_test, y_test)
    evaluate("Random Forest", y_test, rf_preds)

    log_reg_path = os.path.join(model_dir, MODEL_FILES["logistic_regression"])
    joblib.dump(
        {"model": log_reg, "scaler": log_reg.scaler_, "feature_columns": feature_columns},
        log_reg_path,
    )

    rf_path = os.path.join(model_dir, MODEL_FILES["random_forrest"])
    joblib.dump(
        {"model": rf, "scaler": None, "feature_columns": feature_columns},
        rf_path,
    )

    print(f"\nSaved Logistic Regression -> {log_reg_path}")
    print(f"Saved Random Forest       -> {rf_path}")

    return log_reg, rf


if __name__ == "__main__":
    csv_path = os.path.join(SCRIPT_DIR, "..", "data", "market_data.csv")
    df = pd.read_csv(csv_path, parse_dates=["date"])
    train_and_save_models(df)