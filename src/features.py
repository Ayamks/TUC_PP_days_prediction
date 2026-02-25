import pandas as pd


def daily_weather_from_hourly(weather_csv_path: str) -> pd.DataFrame:
    """
    Convert hourly weather data into daily aggregated features.
    Output columns:
      - date (date)
      - is_forecast (bool)
      - temp_min, temp_max, temp_mean
      - humidity_level
    """
    w = pd.read_csv(weather_csv_path)

    # hourly timestamp -> datetime
    w["date"] = pd.to_datetime(w["date"])
    # day key
    w["day"] = w["date"].dt.date

    daily = (
        w.groupby(["day", "is_forecast"], as_index=False)
        .agg(
            temp_min=("temperature", "min"),
            temp_max=("temperature", "max"),
            temp_mean=("temperature", "mean"),
            humidity_level=("relative_humidity", "mean"),
        )
        .rename(columns={"day": "date"})
    )

    # Ensure proper types
    daily["is_forecast"] = daily["is_forecast"].astype(bool)

    daily = add_calendar_features(daily)
    return daily




def load_signals(signals_csv_path: str) -> pd.DataFrame:
    """
    Load PP1/PP2 signals history.
    Output columns:
      - date (date)
      - pp1 (bool)
      - pp2 (bool)
    """
    s = pd.read_csv(signals_csv_path)
    s["date"] = pd.to_datetime(s["date"]).dt.date

    s = s[["date", "PP1", "PP2"]].rename(columns={"PP1": "pp1", "PP2": "pp2"})
    s["pp1"] = s["pp1"].astype(bool)
    s["pp2"] = s["pp2"].astype(bool)
    return s


def build_training_table(signals_df: pd.DataFrame, daily_weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge observed (non-forecast) daily weather with PP signals to create a training dataset.
    """
    # Train only on observed weather to avoid mixing with forecast distributions
    obs_weather = daily_weather_df[daily_weather_df["is_forecast"] == False].copy()

    # Inner join: keep only dates that exist in both datasets
    train = obs_weather.merge(signals_df, on="date", how="inner")

    # basic cleaning: drop rows with missing values in essential columns
    train = train.dropna(subset=["temp_min", "temp_max", "humidity_level", "pp1", "pp2"])

    return train



def add_calendar_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-related features from the 'date' column.
    Expects daily_df['date'] to be python date objects.
    """
    out = daily_df.copy()
    dt = pd.to_datetime(out["date"])

    out["day_of_year"] = dt.dt.dayofyear
    out["month"] = dt.dt.month
    out["is_weekend"] = (dt.dt.weekday >= 5).astype(int)

    # Energy-relevant feature: Heating Degree Days (base 18°C)
    out["hdd_18"] = (18.0 - out["temp_mean"]).clip(lower=0.0)

    return out
