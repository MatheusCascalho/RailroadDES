from dataclasses import dataclass, asdict
from src.domain.constants import EPSILON


@dataclass(frozen=True)
class Flow:
    origin: str
    destination: str
    product: str


@dataclass
class Demand:
    flow: Flow
    volume: float
    operated: float = 0.0
    promised: float = 0.0

    def __post_init__(self):
        if isinstance(self.flow, dict):
            self.flow = Flow(**self.flow)

    @property
    def is_completed(self):
        return self.volume - self.operated < EPSILON

    @property
    def cut(self):
        return self.volume - self.operated

    def to_json(self):
        return {
            "flow": asdict(self.flow),
            "volume": self.volume
        }
