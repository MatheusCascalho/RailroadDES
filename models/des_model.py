import abc
from des_simulator import DESSimulator
import model_queue as mq
from dataclasses import dataclass, field, InitVar
from typing import Any, Generator, Callable
from datetime import timedelta, datetime
from entities import Node, Train
from event_calendar import Event
from models.conditions import RailroadMesh



class DESModel(abc.ABC):
    def __init__(
            self,
            controllable_events: list[Event],
            uncontrollable_events: list[Event],
    ):
        self.controllable_events = []
        self.uncontrollable_events = []

    @abc.abstractmethod
    def starting_events(self, simulator: DESSimulator):
        pass


class Railroad(DESModel):
    def __init__(self, mesh: RailroadMesh, trains: list[Train]):
        super().__init__(
            controllable_events=[],
            uncontrollable_events=[]
        )
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
