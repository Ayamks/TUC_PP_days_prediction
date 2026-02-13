from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class WeatherRecord:
    date: date
    temp_min: float
    temp_max: float
    humidity_level: float
    ...  # Add more fields as necessary
