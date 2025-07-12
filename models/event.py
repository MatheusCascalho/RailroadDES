from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Any


@dataclass
class Event:
    time_until_happen: timedelta
    callback: Callable
    data: Any

    def __repr__(self):
        e = self.callback.__qualname__
        t = self.time_until_happen
        return f"{t} - {e}"

    __str__ = __repr__

    def reschedule(self, time_to_happen):
        self.time_until_happen = time_to_happen


class EventFactory(ABC):
    @abstractmethod
    def create(self, time_until_happen, callback, data):
        pass


class DefaultEventFactory(EventFactory):
    def create(self, time_until_happen, callback, data):
        return Event(time_until_happen, callback, data)
