from dataclasses import dataclass
from models.constants import EPSILON


@dataclass
class Flow:
    origin: str
    destination: str
    product: str


@dataclass
class Demand:
    flow: Flow
    volume: float
    operated: float = 0.0

    @property
    def is_completed(self):
        return self.volume - self.operated < EPSILON

    @property
    def cut(self):
        return self.volume - self.operated
