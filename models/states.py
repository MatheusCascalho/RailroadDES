from dataclasses import dataclass, field
from typing import Any
from datetime import timedelta
from models.constants import TrainActions, EPSILON
from models.time_table import TimeRegister
from enum import Enum


class LoadState(Enum):
    LOADING = "Loading"
    LOADED = "Loaded"
    UNLOADING = "Unloading"
    EMPTY = "Empty"


class ActivityState(Enum):
    MOVING = "Moving"
    QUEUE_TO_ENTER = "Queue to enter"
    PROCESSING = "Processing"
    QUEUE_TO_LEAVE = "Queue to leave"


class ProcessorState(Enum):
    IDLE = "Idle"
    BUSY = "Busy"


class NodeProcessState(Enum):
    READY = "Ready"
    BUSY = "Busy"
    BLOCKED = "Blocked"


@dataclass
class TrainState:
    current_location: Any
    action: TrainActions
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
