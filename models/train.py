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
    NodeState,
    LoadState,
    ActivityState
)
from models.time_table import TimeRegister
from models.resources import Slot
from models.entities import Entity
from interfaces.train_interface import TrainInterface
from interfaces.node_interce import NodeInterface
from collections import defaultdict
from models.demand import Demand
from models.state_machine import StateMachine, Transition, State
from models.task import Task
from models.time_table import EventName, TimeEvent


class LoadSystem:
    def __init__(
            self,
            capacity: float,
            is_loaded: bool,
            processing_state: State,
            queue_to_leave_state: State
    ):
        self.capacity = capacity
        self.state_machine = self.build_state_machine(
            processing_state=processing_state,
            queue_to_leave_state=queue_to_leave_state,
            is_loaded=is_loaded
        )
        self.volume = capacity if is_loaded else 0

    def __str__(self):
        return str(self.state_machine.current_state)

    __repr__ = __str__
    def build_state_machine(self, processing_state, queue_to_leave_state, is_loaded):
        loaded = State(name=LoadState.LOADED, is_marked=is_loaded)
        loading = State(name=LoadState.LOADING, is_marked=False)
        empty = State(name=LoadState.EMPTY, is_marked=not is_loaded)
        unloading = State(name=LoadState.UNLOADING, is_marked=False)
        to_loaded = Transition(
            name="to_loaded",
            origin=loading,
            destination=loaded,
            action=self.load
        )
        to_unloading = Transition(
            name="to_unloading",
            origin=loaded,
            destination=unloading
        )
        to_empty = Transition(
            name="to_empty",
            origin=unloading,
            destination=empty,
            action=self.unload
        )
        to_loading = Transition(
            name="to_loading",
            origin=empty,
            destination=loading
        )

        processing_state.add_observer([to_loading, to_unloading])
        queue_to_leave_state.add_observer([to_loaded, to_empty])
        transitions = [to_loaded, to_empty, to_loading, to_unloading]
        sm = StateMachine(transitions=transitions)
        return sm

    def set_processing_state(self, processing_state: State):
        loading = self.state_machine.transitions['to_loading']
        unloading = self.state_machine.transitions['to_unloading']
        processing_state.add_observer([loading, unloading])

    def set_in_queue_state(self, in_queue_state: State):
        loaded = self.state_machine.transitions['to_loaded']
        empty = self.state_machine.transitions['to_empty']
        in_queue_state.add_observer([loaded, empty])

    def load(self):
        self.volume = self.capacity

    def unload(self):
        self.volume = 0

class ActivitySystem:
    def __init__(self, path):
        self.state_machine = self.build_transitions()
        self.__path = path
        self.current_location = path[0]
        print(self)

    def __str__(self):
        return str(self.state_machine.current_state)

    __repr__ = __str__

    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, new_path):
        self.__path = new_path
        self.current_location = self.__path[0]

    @staticmethod
    def build_transitions():
        moving = State(name=ActivityState.MOVING, is_marked=False)
        queue_to_enter = State(name=ActivityState.QUEUE_TO_ENTER, is_marked=True)
        processing = State(name=ActivityState.PROCESSING, is_marked=False)
        queue_to_leave = State(name=ActivityState.QUEUE_TO_LEAVE, is_marked=False)

        leave = Transition(
            name="leave",
            origin=queue_to_leave,
            destination=moving
        )
        arrive = Transition(
            name="arrive",
            origin=moving,
            destination=queue_to_enter
        )
        start = Transition(
            name="start",
            origin=queue_to_enter,
            destination=processing
        )
        finish = Transition(
            name="finish",
            origin=processing,
            destination=queue_to_leave
        )
        transitions = [
            leave, arrive,
            start, finish
        ]
        sm = StateMachine(transitions=transitions)
        return sm

    @property
    def processing_state(self):
        return self.state_machine.states[ActivityState.PROCESSING]


    @property
    def queue_to_leave_state(self):
        return self.state_machine.states[ActivityState.QUEUE_TO_LEAVE]


    def finish_process(self):
        if self.state_machine.current_state.name == ActivityState.PROCESSING:
            self.state_machine.update()

    def start_process(self):
        if self.state_machine.current_state.name == ActivityState.QUEUE_TO_ENTER:
            self.state_machine.update()

    def leave(self):
        if self.state_machine.current_state.name == ActivityState.QUEUE_TO_LEAVE:
            self.state_machine.update()

    def arrive(self):
        if self.state_machine.current_state.name == ActivityState.MOVING:
            self.state_machine.update()
            self.path = self.path[1:]

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