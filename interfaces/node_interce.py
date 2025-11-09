from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from src.entity import Entity
from src.clock import Clock


class NodeInterface(Entity, ABC):
    def __init__(self, name, clock):
        super().__init__(name=name,clock=clock, prefix='Node')

    @property
    @abstractmethod
    def constraints(self):
        pass

    # ====== Properties ==========
    # ====== Events ==========
    @abstractmethod
    def receive(self, **kwargs):
        pass

    @abstractmethod
    def process(self, **kwargs):
        pass

    @abstractmethod
    def maneuver_to_dispatch(self, **kwargs):
        pass

    @abstractmethod
    def to_json(self):
        pass

    def pos_processing(self):
        pass

    def pre_processing(self):
        pass

    @property
    @abstractmethod
    def daily_capacity(self):
        pass

    @abstractmethod
    def process_time(self, train_size) -> timedelta:
        pass

