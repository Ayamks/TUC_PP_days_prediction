---

# My Implementation

## Overview

This implementation builds a daily classification model to predict whether a given day will be flagged as **PP1**, **PP2**, both, or neither, based on weather data and calendar features.

The solution follows a simple, production-oriented approach:

- Daily aggregation of hourly weather data
- Time-aware training split
- Two independent binary classifiers (PP1 and PP2)
- Imbalance handling
- Dockerized inference pipeline

---

## Data Processing

### Weather Aggregation

The provided weather dataset contains hourly observations.  
Since PP signals are daily, weather data is aggregated per day:

- `temp_min`
- `temp_max`
- `temp_mean`
- `humidity_level` (daily average)

Only observed weather data (`is_forecast=False`) is used for training.

### Calendar Features

To capture seasonality and consumption patterns, the following features were added:

- `day_of_year`
- `month`
- `is_weekend`
- `hdd_18` (Heating Degree Days):  
  `max(0, 18 - temp_mean)`

The HDD feature is particularly relevant as electricity demand in France is strongly correlated with heating during cold periods.

---

## Modeling Strategy

### Two Independent Binary Classifiers

Instead of framing the task as a 4-class problem, two independent models were trained:

- One for PP1
- One for PP2

This design choice was made because:

- PP1 and PP2 signals are independent
- A multi-class formulation would worsen class imbalance
- Binary classification improves interpretability and robustness

### Model Choice

A `RandomForestClassifier` was used with:

- `class_weight="balanced"` to address class imbalance
- 300 trees
- Default threshold = 0.5

Random Forest was chosen because:

- It captures non-linear relationships
- It handles feature interactions
- It requires minimal preprocessing
- It is robust and interpretable

---

## Time-Based Validation

A temporal split was applied to avoid data leakage:

- Training set: years ≤ 2024
- Test set: years ≥ 2025

This setup better reflects real-world forecasting conditions.

---

## Class Imbalance

PP days are rare events:

- ~4% for PP1
- ~7% for PP2

Therefore, accuracy alone is not a reliable metric.  
F1-score and recall were considered more informative for evaluation.

---

## Inference Pipeline

The prediction pipeline:

1. Aggregates hourly weather data to daily level
2. Recreates all training features
3. Filters forecast data
4. Applies trained models
5. Outputs predictions for the requested date range

Output columns:

- `date`
- `temp_min`
- `temp_max`
- `humidity_level`
- `predicted_pp1`
- `predicted_pp2`
- `confidence`

---

## Dockerization

The project is fully Dockerized.

The Docker image:

- Installs dependencies
- Trains models at build time
- Runs prediction via CLI

Build:

```bash
docker build -t pp-predictor .


Run : 

 ```bash
docker run --rm \
  -v "$(pwd)/outputs:/app/outputs" \
  pp-predictor \
  --input data/paris_weather.csv \
  --output outputs/predictions_docker.csv \
  --start 2026-02-20 \
  --end 2026-02-26
