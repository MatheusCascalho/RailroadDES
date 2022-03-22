"""
Entidades do sistema.

Uma entidade é um objeto que realiza/sofre eventos. Toda entidade possui um Estado
"""

import abc
from des_simulator import DESSimulator
import model_queue as mq
from dataclasses import dataclass, field, InitVar
from typing import Any, Generator
from datetime import timedelta, datetime


@dataclass
class Entity:
    state: Any


@dataclass
class Train(Entity):
    id: int
    origin: int
    destination: int
    model: int
    current_location: int
    eta: datetime
    etd: datetime
    start_process: datetime
    finish_process: datetime
    leave: datetime
    time_blocked: timedelta
    volume: float
    path: list[int]

    @property
    def is_empty(self):
        return self.volume <= 1e-1

    def arrive(self):
        self.current_location = self.path.pop()

    @property
    def next_location(self):
        try:
            return self.path[0]
        except:
            raise Exception("Path finished!")


class Node:
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
