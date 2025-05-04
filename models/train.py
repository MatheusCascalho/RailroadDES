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
from models.clock import Clock
from models.states import ActivityState


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
            is_loaded: bool,
            clock: Clock,
            initial_activity: ActivityState = ActivityState.MOVING
    ):
        self.ID = next(train_id)
        self.clock = clock
        self.activity_system = ActivitySystem(
            path=task.path,
            initial_activity=initial_activity
        )
        self.load_system = LoadSystem(
            capacity=capacity,
            is_loaded=is_loaded,
            processing_state=self.activity_system.processing_state,
            queue_to_leave_state=self.activity_system.queue_to_leave_state
        )
        self.__current_task = task
        self._in_slot = False


    # ====== Properties ==========
    @property
    def ID(self):
        return self._train_id

    @ID.setter
    def ID(self, value):
        self._train_id = value

    @property
    def current_task(self):
        return self.__current_task

    @current_task.setter
    def current_task(self, task: Task):
        self.__current_task = task

    @property
    def is_empty(self):
        return self.load_system.volume <= EPSILON

    @property
    def next_location(self):
        try:
            return self.activity_system.path.next_location()
        except IndexError:
            TrainExceptions.path_is_finished()

    @property
    def volume(self):
        return self.load_system.volume

    @property
    def capacity(self):
        return self.load_system.capacity

    @property
    def product(self):
        return self.current_task.demand.flow.product

    @volume.setter
    def volume(self, new_volume):
        self.load_system.volume = new_volume

    @property
    def current_location(self):
        return self.activity_system.path.current_location

    @property
    def process_end(self):
        return self.current_task.time_table.process_end

    @property
    def process(self) -> Callable:
        return self.start_load if self.is_empty else self.start_unload

    @property
    def current_process_name(self):
        return Process.LOAD if self.is_empty else Process.UNLOAD

    @property
    def state(self):
        s = f"Atividade={self.activity_system} | Carga={self.load_system}"
        return s

    @property
    def ready_to_leave(self):
        return self.load_system.is_ready

    def __str__(self):
        name = self.ID
        return f"{name} | Capacidade: {self.capacity} | {self.state}"

    def add_to_slot(self):
        self._in_slot = True

    def removed_from_slot(self):
        self._in_slot = False

    __repr__ = __str__

    # ====== Properties ==========
    # ====== Events ==========

    def finish_load(
        self,
        simulator: DESSimulator,
        node: NodeInterface,
        **kwargs
    ):
        print(f'{self.clock.current_time}:: Train {self.ID} finish load!')
        self.activity_system.finish_process()
        event = TimeEvent(
            event=EventName.FINISH_PROCESS,
            instant=self.clock.current_time
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
        if not self._in_slot:
            TrainExceptions.train_is_not_in_slot(train_id=self.ID, location=self.current_location)
        print(f'{self.clock.current_time}:: Train {self.ID} start load!')
        self.activity_system.start_process()
        event = TimeEvent(
            event=EventName.START_PROCESS,
            instant=self.clock.current_time
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
        if not self._in_slot:
            raise Exception("Train is not in slot!")
        print(f'{self.clock.current_time}:: Train unloading!')
        # Changing State
        self.activity_system.start_process()
        event = TimeEvent(
            event=EventName.START_PROCESS,
            instant=self.clock.current_time
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
        print(f'{self.clock.current_time}:: Train {self.ID} finish load!')
        self.activity_system.finish_process()
        event = TimeEvent(
            event=EventName.FINISH_PROCESS,
            instant=self.clock.current_time
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

        time = self.clock.current_time + node.time_to_call
        simulator.add_event(
            time=time,
            callback=None#node.
        )

    def maneuvering_to_leave(self):
        self.state.action = TrainActions.MANEUVERING_TO_LEAVE

    def arrive(self, simulator: DESSimulator, node: NodeInterface):
        print(f'{self.clock.current_time}:: train {self.ID} arrive at node {node}!!')
        # Changing State
        self.activity_system.arrive()
        event = TimeEvent(
            EventName.ARRIVE,
            instant=self.clock.current_time
        )
        process = Process.LOAD if self.current_task.is_on_load_point() else Process.UNLOAD
        self.current_task.update(event=event,process=process)

    def leave(self, node: NodeInterface):
        print(f'{self.clock.current_time}:: Train {self.ID} leaving node {node}!')
        self.activity_system.leave()
        event = TimeEvent(
            EventName.DEPARTURE,
            instant=self.clock.current_time
        )
        process = Process.LOAD if self.current_task.is_on_load_point() else Process.UNLOAD
        self.current_task.update(event=event, process=process)
