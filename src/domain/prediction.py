from dataclasses import dataclass

from .weather_record import WeatherRecord


@dataclass(frozen=True)
class Prediction:
    weather_record: WeatherRecord
    predicted_pp1: bool
    predicted_pp2: bool
    confidence: float
