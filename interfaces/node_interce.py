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

    @abstractmethod
    def next_idle_slot(self, current_time):
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
    def time_to_call(self, **kwargs):
        pass

    @abstractmethod
    def connect_neighbor(self, **kwargs):
        pass

    @abstractmethod
    def predicted_time(self, current_time):
        pass

