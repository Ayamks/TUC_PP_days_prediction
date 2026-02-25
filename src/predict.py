import joblib
import pandas as pd

from src.features import daily_weather_from_hourly

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


def predict(input_weather_csv: str, output_csv: str,
            start_date="2026-02-20",
            end_date="2026-02-26"):

    # 1️⃣ Load and prepare weather data
    daily_weather = daily_weather_from_hourly(input_weather_csv)

    # 2️⃣ Keep only forecast data
    daily_weather = daily_weather[daily_weather["is_forecast"] == True].copy()

    # 3️⃣ Filter required date range
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()

    daily_weather = daily_weather[
        (daily_weather["date"] >= start) &
        (daily_weather["date"] <= end)
    ].copy()

    if daily_weather.empty:
        raise ValueError("No forecast data found in the requested date range.")

    # 4️⃣ Load trained models
    model_pp1 = joblib.load("models/model_pp1.joblib")
    model_pp2 = joblib.load("models/model_pp2.joblib")

    X = daily_weather[FEATURES]

    # 5️⃣ Predict probabilities
    proba_pp1 = model_pp1.predict_proba(X)[:, 1]
    proba_pp2 = model_pp2.predict_proba(X)[:, 1]

    # 6️⃣ Apply default threshold 0.5
    daily_weather["predicted_pp1"] = (proba_pp1 >= 0.5).astype(int)
    daily_weather["predicted_pp2"] = (proba_pp2 >= 0.5).astype(int)

    # Simple confidence (average of both probabilities)
    daily_weather["confidence"] = (proba_pp1 + proba_pp2) / 2

    # 7️⃣ Save output
    output = daily_weather[[
        "date",
        "temp_min",
        "temp_max",
        "humidity_level",
        "predicted_pp1",
        "predicted_pp2",
        "confidence"
    ]]

    output.to_csv(output_csv, index=False)

    print(f"Predictions saved to {output_csv}")
