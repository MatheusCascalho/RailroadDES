import abc
from des_simulator import DESSimulator
import model_queue as mq
from dataclasses import dataclass, field, InitVar
from typing import Any, Generator
from datetime import timedelta, datetime
from entities import Node, Train


@dataclass
class TransitTime:
    load_origin: int
    load_destination: int
    loaded_time: timedelta
    empty_time: timedelta


@dataclass
class RailroadMesh:
    load_points: tuple[Node]
    unload_points: tuple[Node]
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


class DESModel(abc.ABC):
    @abc.abstractmethod
    def starting_events(self, simulator: DESSimulator):
        pass


class Railroad(DESModel):
    def __init__(self, mesh: RailroadMesh, trains: list[Train]):
        self.mesh = mesh
        self.trains = trains

    def starting_events(self, simulator: DESSimulator):
        for train in self.trains:
            origin = train.current_location

            try:
                destination = train.next_location
            except Exception:
                train.path = self.create_new_path(current_location=origin)
                destination = train.next_location

            time = self.mesh.transit_time(origin=origin, destination=destination)

            simulator.add_event(
                time=time,
                callback=self.on_finish_loaded_path,
                simulator=simulator,
                train=train
            )

    def on_finish_loaded_path(self, simulator, train: Train):
        time = simulator.time + train.time_blocked
        simulator.add_event(
            time=time,
            callback=self.on_unfinish_loading,
            simulator=simulator
        )

    def on_finish_loading(self, simulator, train):
        origin = train.current_location
        destination = train.next_location
        time = simulator.time + self.mesh.transit_time(origin=origin, destination=destination)
        simulator.add_event(
            time=time,
            callback=self.on_finish_loaded_path,
            simulator=simulator,
            train=train
        )
