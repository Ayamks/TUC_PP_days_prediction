from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class PPSignal:
    date: date
    pp1: bool
    pp2: bool
