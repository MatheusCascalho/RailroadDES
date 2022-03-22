from dataclasses import dataclass, field
from typing import Any
from datetime import datetime, timedelta
from models.constants import TrainActions
from exceptions import TrainExceptions


@dataclass
class TrainState:
    current_location: Any
    action: TrainActions
    volume: float = field(default_factory=float)


@dataclass
class TimeRegister:
    arrive: datetime
    start_process: datetime
    finish_process: datetime
    departure: datetime


@dataclass
class RailroadState:
    operated_volume: float
    completed_travels: int
    loaded_trains: int
    empty_trains: int


@dataclass
class NodeState:
    average_time_on_queue_to_enter: timedelta
    average_time_on_queue_to_leave: timedelta
