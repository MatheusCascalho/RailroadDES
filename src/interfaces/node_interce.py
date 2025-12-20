from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from src.events.event_queue import EventQueue
from src.interfaces.entity import Entity
from src.simulation.clock import Clock


class NodeInterface(Entity, ABC):
    def __init__(self, name, clock):
        super().__init__(name=name,clock=clock, prefix='Node')
        self._event_queue = None

    @property
    @abstractmethod
    def constraints(self):
        pass

    @property
    def event_queue(self):
        if self._event_queue is None:
            raise Exception('Event Queue não foi definida!!')
        return self._event_queue
    
    @event_queue.setter
    def event_queue(self, value):
        if not isinstance(value, EventQueue):
            raise Exception(f'event_queue deve ser uma instancia de EventQueue!!')
        self._event_queue = value

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

