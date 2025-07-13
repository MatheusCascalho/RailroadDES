from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Any


@dataclass
class Event:
    time_until_happen: timedelta
    callback: Callable
    data: Any
    event_name: str = ''

    def __repr__(self):
        if self.event_name:
            e = self.event_name
        else:
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


class DecoratedEventFactory(EventFactory):
    def __init__(self, pos_method: Callable, pre_method: Callable):
        self.pos_method = pos_method
        self.pre_method = pre_method

    def wrapper(self, callback: Callable):
        def decorated(*args, **kwargs):
            self.pre_method(*args, **kwargs)
            callback(*args, **kwargs)
            self.pos_method(*args, **kwargs)
        return decorated

    def create(self, time_until_happen, callback, data):
        decorated_callback = self.wrapper(callback)
        return Event(
            time_until_happen,
            decorated_callback,
            data,
            event_name=f"Decorated {callback.__qualname__}"
        )

