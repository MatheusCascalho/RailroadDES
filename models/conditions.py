import abc
from models.des_simulator import DESSimulator
import models.model_queue as mq
from dataclasses import dataclass, field, InitVar
from typing import Any, Generator, Callable
from datetime import timedelta, datetime
from interfaces.node_interce import NodeInterface


@dataclass
class TransitTime:
    load_origin: int
    load_destination: int
    loaded_time: timedelta
    empty_time: timedelta


@dataclass
class RailroadMesh:
    load_points: tuple[NodeInterface]
    unload_points: tuple[NodeInterface]
    transit_times: list[TransitTime]

    def __iter__(self):
        all_points = self.load_points + self.unload_points
        return all_points.__iter__()

    def transit_time(self, origin, destination):
        is_loaded_transit = origin in self.load_points
        for transit in self.transit_times:
            if is_loaded_transit and transit.load_origin == origin and transit.load_destination == destination:
                return transit.loaded_time
            elif not is_loaded_transit and transit.load_origin == destination and transit.load_destination == origin:
                return transit.empty_time

