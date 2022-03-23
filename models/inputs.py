from dataclasses import dataclass


@dataclass
class Demand:
    volume: float
    origin: str
    destination: str