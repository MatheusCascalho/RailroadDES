from models.des_simulator import DESSimulator
from models.constants import (
    EPSILON,
    TrainActions,
    Process, EventName
)
from typing import Callable
from datetime import timedelta, datetime
from models.time_table import TimeTable
from models.discrete_event_system import LoadSystem, ActivitySystem
from models.exceptions import TrainExceptions, FinishedTravelException
from models.resources import Slot
from interfaces.train_interface import TrainInterface
from interfaces.node_interce import NodeInterface
from models.task import Task
from models.time_table import TimeEvent
from models.clock import Clock
from models.states import ActivityState
from models.observers import to_notify
from logging import debug


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
            path=[],
            initial_activity=initial_activity
        )
        self.load_system = LoadSystem(
            capacity=capacity,
            is_loaded=is_loaded,
            processing_state=self.activity_system.processing_state,
            queue_to_leave_state=self.activity_system.queue_to_leave_state
        )
        self.time_table = TimeTable()
        self.current_task = task
        self._in_slot = False
        super().__init__()


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
        if task:
            # task.scheduler.train = self
            task.assign(self.ID)
            self.activity_system.path = task.path
            self.__current_task = task
        else:
            self.__current_task = task


    @property
    def is_empty(self):
        return self.load_system.volume <= EPSILON

    @property
    def next_location(self):
        try:
            return self.activity_system.path.next_location()
        except IndexError:
            TrainExceptions.path_is_finished(train_id=self.ID, location=self.current_location)

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

    @property
    def dispatched_just_now(self):
        if not self.current_task:
            return False
        return self.current_task.time_table.dispatched_just_now

    @property
    def arrived_right_now(self):
        if not self.current_task:
            return False
        return self.current_task.time_table.arrived_right_now

    @property
    def current_activity(self):
        return self.activity_system.state_machine.current_state


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
        debug(f'{self.clock.current_time}:: Train {self} finish load!')
        self.activity_system.finish_process()
        event = TimeEvent(
            event=EventName.FINISH_PROCESS,
            instant=self.clock.current_time
        )
        self.current_task.update(
            event=event,
            process=Process.LOAD,
            location=None
        )
        self.time_table.update(event, process=Process.LOAD, location=None)
        # Add next event to calendar
        node.maneuver_to_dispatch(simulator)

    def start_load(
        self,
        simulator: DESSimulator,
        process_time: timedelta,
        **kwargs
    ):
        if not self._in_slot:
            TrainExceptions.train_is_not_in_slot(train_id=self.ID, location=self.current_location)
        debug(f'{self.clock.current_time}:: Train {self.ID} start load!')
        self.activity_system.start_process()
        event = TimeEvent(
            event=EventName.START_PROCESS,
            instant=self.clock.current_time
        )
        self.current_task.update(
            event=event,
            process=Process.LOAD,
            location=None
        )
        self.time_table.update(event, process=Process.LOAD)

        # Add next event to calendar
        simulator.add_event(
            time=process_time,
            callback=self.finish_load,
            simulator=simulator,
            node=kwargs.get('node')
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
        debug(f'{self.clock.current_time}:: Train unloading!')
        # Changing State
        self.activity_system.start_process()
        event = TimeEvent(
            event=EventName.START_PROCESS,
            instant=self.clock.current_time
        )
        self.current_task.update(
            event=event,
            process=Process.UNLOAD,
            location=None
        )
        self.time_table.update(event, process=Process.UNLOAD, location=None)

        # Add next event to calendar
        simulator.add_event(
            time=process_time,
            callback=self.finish_unload,
            simulator=simulator,
            node=node,
            slot=slot
        )

    def finish_unload(
        self,
        simulator: DESSimulator,
        node: NodeInterface,
        slot: Slot,
        **kwargs
    ):
        self.activity_system.finish_process()
        event = TimeEvent(
            event=EventName.FINISH_PROCESS,
            instant=self.clock.current_time
        )
        self.current_task.update(
            event=event,
            process=Process.UNLOAD,
            location=None
        )
        self.time_table.update(event, process=Process.UNLOAD, location=None)

        debug(f'{self.clock.current_time}:: Train {self} finish unload!')

        # Add next event to calendar
        simulator.add_event(
            time=timedelta(),
            callback=node.maneuver_to_dispatch,
            simulator=simulator,
            # slot=slot,
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

    @to_notify()
    def arrive(self, node: NodeInterface):
        debug(f'{self.clock.current_time}:: train {self.ID} arrive at node {node}!!')
        # Changing State
        self.activity_system.arrive()
        event = TimeEvent(
            EventName.ARRIVE,
            instant=self.clock.current_time
        )
        process = Process.LOAD if self.current_task.is_on_load_point() else Process.UNLOAD
        self.current_task.update(event=event,process=process, location=node.identifier)
        self.time_table.update(event, process=process, location=node.identifier)

    @to_notify()
    def leave(self, node: NodeInterface):
        debug(f'{self.clock.current_time}:: Train {self.ID} leaving node {node}!')
        self.activity_system.leave()
        event = TimeEvent(
            EventName.DEPARTURE,
            instant=self.clock.current_time
        )
        process = Process.LOAD if self.current_task.is_on_load_point() else Process.UNLOAD
        self.current_task.update(event=event, process=process, location=None)
        self.time_table.update(event, process=process, location=node.identifier)

