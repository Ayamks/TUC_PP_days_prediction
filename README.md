# PP1/PP2 Days Prediction - Technical Case Study

## Context

In France, **PP1** and **PP2** days are special signals activated by **RTE** (the national electricity transmission system operator) to flag periods of high tension on the electrical grid. These days typically occur during intense cold spells or energy consumption peaks. Their purpose is to help reduce energy consumption through specific incentives.

Your goal is to build a model capable of predicting whether a given day will be classified as PP1 and/or PP2, using weather forecast data and the day of the year. This would allow anticipating periods of grid tension.

RTE publishes the signal **the day before** (D-1): PP1 and PP2-on-consumption-criteria at 9:30 AM, PP2-on-grid-tension-criteria by 7:00 PM at the latest.

> To learn more about PP1/PP2 signals: [RTE documentation](https://www.services-rte.com/fr/visualisez-les-donnees-publiees-par-rte/signaux-pp1-et-pp2.html)

## Objectives

1. **Build a classification model** in Python to predict whether a day is PP1, PP2, both, or neither.
2. **Provide a Docker image** to run the model. The image must:
   - Accept an input CSV file with weather data (`date`, `temp_min`, `temp_max`, `humidity_level`)
   - Generate an output CSV file with additional columns indicating the prediction for each day
3. **Predict** whether each day from **February 20 to February 26, 2026** (inclusive) will be PP1 and/or PP2. Weather forecast data for this period is available in the dataset (`is_forecast=True`).
4. **Explore** (if relevant) adding supplementary weather features to improve predictions.

## Data

Two datasets are provided in `data/`:

### `pp1_pp2_signals.csv` — RTE PP signals history

Daily records from **2022-01-01 to 2026-02-11** (~1,500 rows).

| Column | Type | Description |
|--------|------|-------------|
| `date` | `YYYY-MM-DD` | Calendar date |
| `PP1` | `bool` | Whether the day was flagged as PP1 |
| `PP1_updated_date` | `datetime` | Timestamp of the last PP1 signal update |
| `PP2` | `bool` | Whether the day was flagged as PP2 |
| `PP2_updated_date` | `datetime` | Timestamp of the last PP2 signal update |

Note: PP1 and PP2 are **independent** — a day can be both PP1 and PP2 simultaneously.

### `paris_weather.csv` — Hourly weather observations (Paris)

**Hourly** records from **2022-01-01 to 2026-02-26** (~36,400 rows). Includes forecast data for recent dates (`is_forecast=True`).

| Column | Type | Description |
|--------|------|-------------|
| `date` | `datetime` | Timestamp (hourly) |
| `temperature` | `float` | Temperature (°C) |
| `apparent_temperature` | `float` | Feels-like temperature (°C) |
| `precipitation` | `float` | Total precipitation (mm) |
| `rain` | `float` | Rain amount (mm) |
| `shortwave_radiation` | `float` | Solar radiation (W/m²) |
| `cloud_cover` | `float` | Cloud cover (%) |
| `dew_point` | `float` | Dew point temperature (°C) |
| `relative_humidity` | `float` | Relative humidity (%) |
| `wind_speed` | `float` | Wind speed (km/h) |
| `coordinates` | `string` | Lat/lon of the station |
| `elevation` | `string` | Station elevation |
| `timezone` | `string` | Timezone |
| `utc_offset` | `string` | UTC offset |
| `is_forecast` | `bool` | Whether this row is forecast vs. observed |
| `updated_at` | `datetime` | Last update timestamp |

## Project Structure

```
.
├── README.md
├── data/
│   ├── pp1_pp2_signals.csv
│   └── paris_weather.csv
├── src/
│   ├── domain/
│   │   ├── weather_record.py        # Daily aggregated weather observation
│   │   ├── pp_signal.py             # PP signal emitted by RTE
│   │   └── prediction.py            # Model prediction result
│   └── ...                          # Your code here
├── Dockerfile
└── requirements.txt
```

## Deliverable

- A **private GitHub repository** with clear documentation.
- The evaluation centers on:
  - Your **design choices** and ability to explain them
  - **Code quality** and structure
  - **Model accuracy** and relevance of your approach
  - Skills demonstrated: data manipulation, machine learning, Docker usage

## Getting Started

1. Fork this repository
2. Explore the datasets in `data/` to understand the raw data
3. Look at the domain models in `src/domain/` to understand the target data model
4. Build your solution!
