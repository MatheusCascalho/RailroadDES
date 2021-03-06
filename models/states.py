from dataclasses import dataclass, field
from typing import Any
from datetime import datetime, timedelta
from models.constants import TrainActions, Process, EPSILON
from models.exceptions import TrainExceptions


@dataclass
class TimeRegister:
    process: Process = field(default=None)
    arrive: datetime = field(default=None)
    start_process: datetime = field(default=None)
    finish_process: datetime = field(default=None)
    departure: datetime = field(default=None)


@dataclass
class TrainState:
    current_location: Any
    action: TrainActions
    time_register: TimeRegister
    volume: float = field(default_factory=float)


@dataclass
class RailroadState:
    operated_volume: float
    completed_travels: int
    loaded_trains: int
    empty_trains: int
    target_volume: float

    @property
    def is_incomplete(self):
        return self.target_volume - self.operated_volume > EPSILON


@dataclass
class NodeState:
    average_time_on_queue_to_enter: timedelta = timedelta()
    average_time_on_queue_to_leave: timedelta = timedelta()
    trains_on_queue_to_enter: int = 0
    trains_on_queue_to_leave: int = 0
    trains_on_process: int = 0
