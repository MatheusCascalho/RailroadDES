"""
Entidades do sistema.

Uma entidade é um objeto que realiza/sofre eventos. Toda entidade possui um Estado
"""

import abc
from models.des_simulator import DESSimulator
from models.constants import (
    EPSILON
)
import models.model_queue as mq
from dataclasses import dataclass, field, InitVar
from typing import Any, Generator
from datetime import timedelta, datetime
from models.exceptions import TrainExceptions


@dataclass
class Entity:
    state: Any = field(init=False)


@dataclass
class Train(Entity):
    id: int
    origin: int
    destination: int
    model: int
    current_location: Any
    eta: datetime
    etd: datetime
    volume: float
    path: list[int]
    start_process: datetime = field(init=False, default=None)
    finish_process: datetime = field(init=False, default=None)
    leave_time: datetime = field(init=False, default=None)
    time_blocked: timedelta = field(init=False, default=None)

    @property
    def is_empty(self):
        return self.volume <= EPSILON

    def arrive(self):
        self.current_location = self.path.pop(0)

    def leave(self):
        self.current_location = (self.current_location, self.next_location)

    @property
    def next_location(self):
        try:
            return self.path[0]
        except IndexError:
            TrainExceptions.path_is_finished()


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
