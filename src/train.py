import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src.features import daily_weather_from_hourly, load_signals, build_training_table

# Features used by the model
FEATURES = [
    "temp_min",
    "temp_max",
    "temp_mean",
    "humidity_level",
    "day_of_year",
    "month",
    "is_weekend",
    "hdd_18",
]


def temporal_split(df: pd.DataFrame):
    """
    Time-based split to avoid leakage.
    Train: <= 2024
    Test:  >= 2025
    """
    dt = pd.to_datetime(df["date"])
    train_df = df[dt.dt.year <= 2024].copy()
    test_df = df[dt.dt.year >= 2025].copy()
    return train_df, test_df


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Simple robust baseline model for imbalanced classification.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def main():
    # 1) Load & prepare data
    signals = load_signals("data/pp1_pp2_signals.csv")
    weather_daily = daily_weather_from_hourly("data/paris_weather.csv")
    df = build_training_table(signals, weather_daily)

    # 2) Split
    train_df, test_df = temporal_split(df)

    X_train = train_df[FEATURES]
    X_test = test_df[FEATURES] if len(test_df) > 0 else None

    y_pp1_train = train_df["pp1"].astype(int)
    y_pp2_train = train_df["pp2"].astype(int)

    # 3) Train two independent models
    model_pp1 = train_model(X_train, y_pp1_train)
    model_pp2 = train_model(X_train, y_pp2_train)

    # 4) Evaluate on test (if available)
    if X_test is not None and len(test_df) > 0:
        y_pp1_test = test_df["pp1"].astype(int)
        y_pp2_test = test_df["pp2"].astype(int)

        print("=== PP1 classification report (test set) ===")
        print(classification_report(y_pp1_test, model_pp1.predict(X_test), digits=3))

        print("=== PP2 classification report (test set) ===")
        print(classification_report(y_pp2_test, model_pp2.predict(X_test), digits=3))
    else:
        print("No test data found for year >= 2025. Skipping evaluation.")

    # 5) Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(model_pp1, "models/model_pp1.joblib")
    joblib.dump(model_pp2, "models/model_pp2.joblib")
    print("Saved models to models/ (model_pp1.joblib, model_pp2.joblib)")


if __name__ == "__main__":
    main()
