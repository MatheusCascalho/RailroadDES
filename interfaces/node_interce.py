from abc import ABC, abstractmethod
from dataclasses import dataclass
from models.entity import Entity
from models.clock import Clock


class NodeInterface(Entity, ABC):
    def __init__(self, name, clock):
        super().__init__(name=name,clock=clock)

    # ====== Properties ==========
    # ====== Events ==========
    @abstractmethod
    def process(self, **kwargs):
        pass

    @abstractmethod
    def maneuver_to_dispatch(self, **kwargs):
        pass


