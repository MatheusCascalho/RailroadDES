from models.des_simulator import DESSimulator
from models.constants import (
    EPSILON,
    TrainActions,
    Process, EventName
)
from typing import Callable
from datetime import timedelta, datetime

from models.discrete_event_system import LoadSystem, ActivitySystem
from models.exceptions import TrainExceptions, FinishedTravelException
from models.resources import Slot
from interfaces.train_interface import TrainInterface
from interfaces.node_interce import NodeInterface
from models.task import Task
from models.time_table import TimeEvent


def train_id_gen():
    i = 0
    while True:
        ID = f"train_{i}"
        yield ID
        i += 1

train_id = train_id_gen()

class Train(TrainInterface):
    def __init__(
            self,
            capacity: float,
            task: Task,
            is_loaded: bool
    ):
        self.ID = next(train_id)

        self.activity_system = ActivitySystem(
            path=task.path
        )
        self.load_system = LoadSystem(
            capacity=capacity,
            is_loaded=is_loaded,
            processing_state=self.activity_system.processing_state,
            queue_to_leave_state=self.activity_system.queue_to_leave_state
        )
        self.__current_task = task


    # ====== Properties ==========
    @property
    def current_task(self):
        return self.__current_task

    @property
    def is_empty(self):
        return self.load_system.volume <= EPSILON

    @property
    def next_location(self):
        try:
            return self.activity_system.path[1]
        except IndexError:
            TrainExceptions.path_is_finished()

    @property
    def volume(self):
        return self.load_system.volume

    @property
    def capacity(self):
        return self.load_system.capacity

    @volume.setter
    def volume(self, new_volume):
        self.load_system.volume = new_volume

    @property
    def current_location(self):
        return self.activity_system.current_location

    @property
    def process_end(self):
        return self.current_task.time_table.process_end

    @property
    def process(self) -> Callable:
        return self.start_load if self.is_empty else self.start_unload

    @property
    def current_process_name(self):
        return Process.LOAD if self.is_empty else Process.UNLOAD

    def __str__(self):
        name = self.ID
        state = f"Atividade={self.activity_system} | Carga={self.load_system}"
        return f"{name} | {state}"

    __repr__ = __str__

    # ====== Properties ==========
    # ====== Events ==========

    def finish_load(
        self,
        simulator: DESSimulator,
        node: NodeInterface,
        **kwargs
    ):
        print(f'{simulator.current_date}:: Train {self.ID} finish load!')
        self.activity_system.finish_process()
        event = TimeEvent(
            event=EventName.FINISH_PROCESS,
            instant=simulator.current_date
        )
        self.current_task.update(
            event=event,
            process=Process.LOAD
        )
        # Add next event to calendar
        simulator.add_event(
            time=timedelta(),
            callback=node.maneuver_to_dispatch,
            simulator=simulator,
        )

    def start_load(
        self,
        simulator: DESSimulator,
        process_time: timedelta,
        **kwargs
    ):
        print(f'{simulator.current_date}:: Train {self.ID} start load!')
        self.activity_system.start_process()
        event = TimeEvent(
            event=EventName.START_PROCESS,
            instant=simulator.current_date
        )
        self.current_task.update(
            event=event,
            process=Process.LOAD
        )
        # Add next event to calendar
        simulator.add_event(
            time=process_time,
            callback=self.finish_load,
            simulator=simulator,
        )

    def start_unload(
        self,
        simulator: DESSimulator,
        start: datetime,
        process_time: timedelta,
        node: NodeInterface,
        slot: Slot
   ):
        print(f'{simulator.current_date}:: Train unloading!')
        # Changing State
        self.activity_system.start_process()
        event = TimeEvent(
            event=EventName.START_PROCESS,
            instant=simulator.current_date
        )
        self.current_task.update(
            event=event,
            process=Process.UNLOAD
        )
        # Add next event to calendar
        simulator.add_event(
            time=process_time,
            callback=self.finish_unload,
            simulator=simulator,
        )

    def finish_unload(
        self,
        simulator: DESSimulator,
        node: NodeInterface,
        slot: Slot,
        **kwargs
    ):
        print(f'{simulator.current_date}:: Train {self.ID} finish load!')
        self.activity_system.finish_process()
        event = TimeEvent(
            event=EventName.FINISH_PROCESS,
            instant=simulator.current_date
        )
        self.current_task.update(
            event=event,
            process=Process.UNLOAD
        )
        # Add next event to calendar
        simulator.add_event(
            time=timedelta(),
            callback=node.maneuver_to_dispatch,
            simulator=simulator,
            slot=slot,
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
        print(f'{simulator.current_date}:: train {self.ID} arrive at node {node}!!')
        # Changing State
        self.activity_system.arrive()
        event = TimeEvent(
            EventName.ARRIVE,
            instant=simulator.current_date
        )
        on_load_point = self.activity_system.current_location == self.current_task.demand.flow.origin
        process = Process.LOAD if on_load_point else Process.UNLOAD
        self.current_task.update(event=event,process=process)

    def leave(self, simulator: DESSimulator, node: NodeInterface):
        print(f'{simulator.current_date}:: Train {self.ID} leaving node {node}!')
        self.activity_system.leave()
        event = TimeEvent(
            EventName.DEPARTURE,
            instant=simulator.current_date
        )
        on_load_point = self.activity_system.current_location == self.current_task.demand.flow.origin
        process = Process.LOAD if on_load_point else Process.UNLOAD
        self.current_task.update(event=event, process=process)

        try:

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