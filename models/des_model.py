import abc
from des_simulator import DESSimulator
import model_queue as mq
from dataclasses import dataclass, field, InitVar
from typing import Any, Generator, Callable
from datetime import timedelta, datetime
from models.entities import Node, Train, Entity
from event_calendar import Event
from models.conditions import RailroadMesh
from models.states import RailroadState
from models.inputs import Demand
from models.exceptions import TrainExceptions


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


class Railroad(DESModel, Entity):
    def __init__(self, mesh: RailroadMesh, trains: list[Train], demands: list[Demand]):
        super().__init__(
            controllable_events=[],
            uncontrollable_events=[]
        )
        self.mesh = mesh
        self.trains = trains
        self.state: RailroadState = RailroadState(
            operated_volume=0,
            completed_travels=0,
            loaded_trains=0,
            empty_trains=0
        )

    # ===== Events =========
    def starting_events(self, simulator: DESSimulator):
        for train in self.trains:
            origin = train.current_location
            try:
                destination = train.next_location
            except TrainExceptions:
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

        time = 0#simulator.time + train.state.time_register.tim
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

    # ===== Events =========
    # ===== Decision Methods =========
    def create_new_path(self, current_location: int):
        return [current_location, 1, 0]
