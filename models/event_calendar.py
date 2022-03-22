import math
from typing import Callable, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Event:
    time_until_happen: float
    callback: Callable
    data: Any


class EventCalendar:
    def __init__(self):
        self.calendar: list[Event] = []

    def push(self, time: float, callback: Callable, **data: Any):
        """
        Add event to calendar ordered by time
        :param time: fire time
        :param callback: callback_function
        :param data: custom calback data.
        """
        event = Event(time_until_happen=time, callback=callback, data=data)
        edge_indexes = [0, len(self.calendar)]
        while edge_indexes[0] != edge_indexes[1]:
            pivot = math.floor((edge_indexes[0]+edge_indexes[1])/2)
            if time >= self.calendar[pivot].time_until_happen:
                edge_indexes[0] = pivot + 1
            else:
                edge_indexes[1] = pivot

        self.calendar.insert(edge_indexes[0], event)

    def pop(self):
        """
        remove nearest event
        :return (float): fire time
        :return (callable): callback function
        """
        return self.calendar.pop(0)

    @property
    def is_empty(self):
        """
        Check wheter calendar is empty
        :return:
        """
        return len(self.calendar) == 0


if __name__ == '__main__':
    print("="*50)
    event_calendar = EventCalendar()
    event_calendar.push(0, lambda: print('Primeira função'))
    event_calendar.push(15, lambda: print('Terceira função'))
    event_calendar.push(2, lambda: print('Segunda função'))
    for event in event_calendar.calendar:
        event.callback()
    print("="*50)
    event_calendar.pop()
    for event in event_calendar.calendar:
        event.callback()
    print("="*50)
    print(event_calendar.is_empty)
    print("=" * 50)
    event_calendar.pop()
    event_calendar.pop()

    print(event_calendar.is_empty)
