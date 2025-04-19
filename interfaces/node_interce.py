from abc import ABC, abstractmethod
from dataclasses import dataclass


class NodeInterface(ABC):
    # ====== Properties ==========
    @property
    @abstractmethod
    def identifier(self):
        pass

    @identifier.setter
    @abstractmethod
    def identifier(self, **kwargs):
        pass

    @property
    @abstractmethod
    def process_time(self):
        pass



    # ====== Properties ==========
    # ====== Events ==========
    @abstractmethod
    def process(self, **kwargs):
        pass

    @abstractmethod
    def maneuver_to_dispatch(self, **kwargs):
        pass

    # ====== Events ==========
    # ====== Methods ==========

    @abstractmethod
    def connect_neighbor(self, **kwargs):
        pass
