"""
Entidades do sistema.

Uma entidade é um objeto que realiza/sofre eventos. Toda entidade possui um Estado
"""

import abc
from models.des_simulator import DESSimulator
from models.constants import (
    EPSILON,
    TrainActions
)
import models.model_queue as mq
from dataclasses import dataclass, field, InitVar
from typing import Any, Generator
from datetime import timedelta, datetime
from models.exceptions import TrainExceptions
from models.states import (
    TrainState,
    TimeRegister
)


@dataclass
class Entity:
    state: Any = field(init=False)


@dataclass
class Train(Entity):
    id: int
    origin: int
    destination: int
    model: int
    path: list[int]
    state: TrainState = field(init=False)
    time_table: dict[int, TimeRegister] = field(init=False)
    initial_volume: InitVar[float] = field(default=0.0)

    def __post_init__(self, initial_volume: float):
        self.state = TrainState(
            volume=initial_volume,
            current_location=self.path.pop(0),
            action=TrainActions.MOVING
        )

    # ====== Properties ==========
    @property
    def is_empty(self):
        return self.state.volume <= EPSILON

    @property
    def next_location(self):
        try:
            return self.path[0]
        except IndexError:
            TrainExceptions.path_is_finished()

    @property
    def volume(self):
        return self.state.volume

    @volume.setter
    def volume(self, new_volume):
        self.volume = new_volume

    # ====== Properties ==========
    # ====== Events ==========
    def load(self, volume, start, end):
        self.volume += volume
        self.state.action = TrainActions.LOADING

    def unload(self, volume):
        if volume > self.volume:
            TrainExceptions.volume_to_unload_is_greater_than_current_volume()
        self.volume -= volume
        self.state.action = TrainActions.UNLOADING

    def maneuvering_to_enter(self):
        self.state.action = TrainActions.MANEUVERING_TO_ENTER

    def maneuvering_to_leave(self):
        self.state.action = TrainActions.MANEUVERING_TO_LEAVE

    def arrive(self):
        self.state.current_location = self.path.pop(0)
        self.state.action = TrainActions.MANEUVERING_TO_ENTER

    def leave(self):
        self.state.current_location = (self.state.current_location, self.next_location)

    # ====== Events ==========


class Node(Entity):
    def __init__(self, queue_capacity: int, identifier: int):
        self._id = identifier
        self.queue: mq.Queue(capacity=queue_capacity)
        self.train_schedule: list[Train] = []

    @property
    def identifier(self):
        return self._id

    @identifier.setter
    def identifier(self, new_identifier: int):
        self._id = new_identifier

    @abc.abstractmethod
    def process_time(self) -> float:
        pass
