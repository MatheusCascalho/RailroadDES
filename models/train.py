import abc
from models.des_simulator import DESSimulator
from models.constants import (
    EPSILON,
    TrainActions,
    Process
)
import models.model_queue as mq
from dataclasses import dataclass, field, InitVar
from typing import Union, Generator, Callable
from datetime import timedelta, datetime
from models.exceptions import TrainExceptions, FinishedTravelException
from models.states import (
    TrainState,
    TimeRegister,
    NodeState
)
from models.resources import Slot
from models.entities import Entity
from interfaces.train_interface import TrainInterface
from interfaces.node_interce import NodeInterface
from collections import defaultdict
from models.demand import Demand


@dataclass
class Train(TrainInterface):

    id: int
    origin: int
    destination: int
    model: int
    path: list[int]
    capacity: float
    target_demand: Demand = field(init=False)
    state: TrainState = field(init=False)
    time_table: dict[int, list[TimeRegister]] = field(init=False, default_factory=lambda : defaultdict(list))
    current_location: InitVar[Union[int, list[int]]]
    initial_volume: InitVar[float] = field(default=0.0)

    def __post_init__(self, current_location, initial_volume: float):
        self.state = TrainState(
            volume=initial_volume,
            current_location=current_location,
            action=TrainActions.WAITING_ON_QUEUE_TO_LEAVE if isinstance(current_location, int) else TrainActions.MOVING,
            time_register=TimeRegister()
        )


    # ====== Properties ==========
    @property
    def is_empty(self):
        return self.state.volume <= EPSILON

    @property
    def next_location(self):
        try:
            return self.path[0]
        except IndexError:
            TrainExceptions.path_is_finished()

    @property
    def volume(self):
        return self.state.volume

    @volume.setter
    def volume(self, new_volume):
        self.state.volume = new_volume

    @property
    def current_location(self):
        return self.state.current_location

    @current_location.setter
    def current_location(self, location):
        self.state.current_location = location

    @property
    def process(self) -> Callable:
        return self.load if self.is_empty else self.unload

    @property
    def action(self):
        return self.state.action
    # ====== Properties ==========
    # ====== Events ==========

    def load(
        self,
        simulator: DESSimulator,
        start: datetime,
        process_time: timedelta,
        node: NodeInterface,
        slot: Slot,
        **kwargs
    ):
        print(f'{simulator.current_date}:: Train {self.id} loading!')
        if self.state.action == TrainActions.MOVING:
            TrainExceptions.processing_when_train_is_moving()
        # Changing State
        self.volume += self.capacity
        self.state.action = TrainActions.LOADING
        self.time_table[self.current_location][-1].process = Process.LOAD
        self.time_table[self.current_location][-1].start_process = start
        self.time_table[self.current_location][-1].finish_process = start + process_time
        self.target_demand.operated += self.capacity

        # Add next event to calendar
        simulator.add_event(
            time=process_time,
            callback=node.maneuver_to_dispatch,
            simulator=simulator,
            slot=slot,
        )

    def unload(
        self,
        simulator: DESSimulator,
        start: datetime,
        process_time: timedelta,
        node: NodeInterface,
        slot: Slot
   ):
        print(f'{simulator.current_date}:: Train unloading!')
        # Changing State
        if self.capacity > self.volume:
            TrainExceptions.volume_to_unload_is_greater_than_current_volume()
        self.volume -= self.capacity
        self.state.action = TrainActions.UNLOADING
        self.time_table[self.current_location][-1].start_process = start
        self.time_table[self.current_location][-1].finish_process = start + process_time
        self.time_table[self.current_location][-1].process = Process.UNLOAD


        # Add next event to calendar
        simulator.add_event(
            time=process_time,
            callback=node.maneuver_to_dispatch,
            simulator=simulator,
            slot=slot
        )

    def maneuvering_to_enter(self, simulator: DESSimulator, node: NodeInterface):
        self.state.action = TrainActions.MANEUVERING_TO_ENTER

        time = simulator.current_date + node.time_to_call
        simulator.add_event(
            time=time,
            callback=None#node.
        )

    def maneuvering_to_leave(self):
        self.state.action = TrainActions.MANEUVERING_TO_LEAVE

    def arrive(self, simulator: DESSimulator, node: NodeInterface):
        print(f'{simulator.current_date}:: train {self.id} arrive at node {node}!!')
        # Changing State
        self.current_location = self.path.pop(0)
        self.state.action = TrainActions.MANEUVERING_TO_ENTER
        self.time_table[self.current_location].append(
            TimeRegister(
                arrive=simulator.current_date
            )
        )

        # Add next event to calendar
        simulator.add_event(
            time=timedelta(),
            callback=node.call_to_enter,
            simulator=simulator,
            train=self,
            arrive=simulator.current_date
        )

    def leave(self, simulator: DESSimulator, node: NodeInterface):
        print(f'{simulator.current_date}:: Train {self.id} leaving node {node}!')
        self.time_table[self.current_location][-1].departure = simulator.current_date
        try:
            self.current_location = [self.current_location, self.next_location]
            self.state.action = TrainActions.MOVING

            simulator.add_event(
                time=node.neighbors[self.next_location].transit_time,
                callback=self.arrive,
                simulator=simulator,
                node=node.neighbors[self.next_location].neighbor
            )
        except IndexError:
            raise FinishedTravelException.path_is_finished(train=self, current_time=simulator.current_date)
        except TrainExceptions as error:
            if error.args[0].lower() == 'path is finished!':
                raise FinishedTravelException.path_is_finished(train=self, current_time=simulator.current_date)
    # ====== Events ==========

    # ====== Statistics ==========
    def loaded_volume(self):
        operated = {
            node: sum(
                self.capacity
                for register in registers
                if register.process == Process.LOAD
            )
            for node, registers in self.time_table.items()
        }
        return operated

    def unloaded_volume(self):
        operated = {
            node: sum(
                self.capacity
                for register in registers
                if register.process == Process.UNLOAD
            )
            for node, registers in self.time_table.items()
        }
        return operated