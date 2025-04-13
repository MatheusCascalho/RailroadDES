from models.entities import Entity
import abc
from models.des_simulator import DESSimulator
from models.constants import (
    EPSILON,
    TrainActions,
    Process
)
from dataclasses import dataclass, field, InitVar
from typing import Any, Generator, Callable
from datetime import timedelta, datetime
from models.exceptions import TrainExceptions
from models.states import (
    TrainState,
    NodeState,
    ProcessorState,
    NodeProcesState
)
from models.time_table import TimeRegister
from models.entities import Entity
from models.railroad import RailSegment
from interfaces.node_interce import NodeInterface
from interfaces.train_interface import TrainInterface
from models.resources import Slot
import models.model_queue as mq
import numpy as np
from models.state_machine import StateMachine, State, Transition, MultiCriteriaTransition
from typing import Union
from models.clock import Clock


class ProcessorSystem:
    def __init__(self, processor_type: Process, queue_to_leave: mq.Queue, clock: Clock):
        self.type = processor_type
        self.state_machine = self.build_state_machine()
        self.current_train: Union[None, TrainInterface] = None
        self.queue_to_leave = queue_to_leave
        self.clock = clock

    def __str__(self):
        return str(self.state_machine.current_state)

    __repr__ = __str__

    def build_state_machine(self):
        idle = State(name=ProcessorState.IDLE, is_marked=True)
        busy = State(name=ProcessorState.BUSY, is_marked=False)
        put_in = Transition(name="Put in", origin=idle, destination=busy)
        free_up = Transition(name="Free Up", origin=busy, destination=idle, action=self.clear)
        sm = StateMachine(transitions=[put_in, free_up])
        return sm

    def put_in(self, train: TrainInterface):
        if self.state_machine.current_state.name == ProcessorState.IDLE:
            self.current_train = train
            self.state_machine.update()
        else:
            raise Exception("Processor is Busy")

    def free_up(self):

        if self.state_machine.current_state.name == ProcessorState.BUSY:
            self.current_train = None
            self.state_machine.update()
        else:
            raise Exception("Processor is Idle")

    def clear(self):
        train = self.current_train
        self.queue_to_leave.push(
            train,
            arrive=self.clock.current_time
        )
        return train

    def is_ready_to_clear(self):
        if self.current_train:
            return self.current_train.process_end <= self.clock.current_time
        return False

    @property
    def is_idle(self):
        return self.state_machine.current_state.name == ProcessorState.IDLE

@dataclass
class Neighbor:
    neighbor: NodeInterface
    transit_time: float

class ProcessConstraintSystem:
    def __init__(
            self,
    ):
        self.state_machine = self.build_state_machine()

    def __str__(self):
        return str(self.state_machine.current_state)

    __repr__ = __str__

    @staticmethod
    def build_state_machine() -> StateMachine:
        ready = State(name=NodeProcesState.READY, is_marked=True)
        busy = State(name=NodeProcesState.BUSY, is_marked=False)
        blocked = State(name=NodeProcesState.BLOCKED, is_marked=False)

        start = MultiCriteriaTransition(
            name="start",
            origin=ready,
            destination=busy
        )
        finish = MultiCriteriaTransition(
            name="finish",
            origin=busy,
            destination=ready
        )
        block = MultiCriteriaTransition(
            name="block",
            origin=ready,
            destination=blocked
        )
        release = MultiCriteriaTransition(
            name="release",
            origin=blocked,
            destination=ready
        )
        sm = StateMachine(transitions=[
            start,finish,
            block,release
        ])
        return sm

    def is_blocked(self):
        return self.state_machine.current_state.name == NodeProcesState.BLOCKED



class Node(NodeInterface):
    def __init__(
            self,
            queue_capacity: int,
            name: Any,
            process_time: timedelta,
            clock: Clock,
            load_constraint_system: ProcessConstraintSystem,
            unload_constraint_system: ProcessConstraintSystem,
            load_units_amount: int = 0,
            unload_units_amount: int = 0
    ):
        self._id = name
        self.name = name
        self.clock = clock
        self.process_constraint = {
            Process.LOAD: load_constraint_system,
            Process.UNLOAD: unload_constraint_system
        }
        self.queue_to_enter = mq.Queue(capacity=queue_capacity)
        self.queue_to_leave = mq.Queue(capacity=float('inf'))
        self.load_units: list[ProcessorSystem] = [
            ProcessorSystem(
                processor_type=Process.LOAD,
                queue_to_leave=self.queue_to_leave,
                clock=clock
            )
            for _ in range(load_units_amount)
        ]
        for unit in self.load_units:
            unit.state_machine.states[ProcessorState.BUSY].add_observer(
                self.process_constraint[Process.LOAD].state_machine.transitions["start"]
            )
            unit.state_machine.states[ProcessorState.IDLE].add_observer(
                self.process_constraint[Process.LOAD].state_machine.transitions["finish"]
            )
        self.unload_units: list[ProcessorSystem] = [
            ProcessorSystem(
                processor_type=Process.UNLOAD,
                queue_to_leave=self.queue_to_leave,
                clock=clock
            )
            for _ in range(unload_units_amount)
        ]
        for unit in self.unload_units:
            unit.state_machine.states[ProcessorState.BUSY].add_observer(
                self.process_constraint[Process.UNLOAD].state_machine.transitions["start"]
            )
            unit.state_machine.states[ProcessorState.IDLE].add_observer(
                self.process_constraint[Process.UNLOAD].state_machine.transitions["finish"]
            )
        self.neighbors: dict[int, RailSegment] = {}
        self._process_time = process_time

    # ====== Events ==========
    def receive(self, train):
        self.queue_to_enter.push(train, arrive=self.clock.current_time)

    def dispatch(self):
        while self.queue_to_leave.is_busy:
            train = self.queue_to_leave.pop(self.clock.current_time)
            next_node = train.next_location
            self.neighbors[next_node].send(train)

    def process(self, simulator: DESSimulator):
        while True:
            # Update resources
            train = self.queue_to_enter.first
            if not train:
                break
            process = train.current_process_name
            if self.process_constraint[process].is_blocked():
                self.queue_to_enter.skip_process(process)
                break
            processors = self.load_units if process == Process.LOAD else self.unload_units
            slot = next((p for p in processors if p.is_idle), None)
            if slot is not None:
                print(f'{simulator.current_date}:: Train {self.queue_to_enter.elements[0].element.ID} starts process at node {self}!')
                train = self.queue_to_enter.pop(
                    current_time=simulator.current_date
                )

                slot.put_in(train=train)

                # Update state
                time = timedelta()
                # Add next event
                simulator.add_event(
                    time=time,
                    callback=train.process,
                    simulator=simulator,
                    start=simulator.current_date,
                    process_time=self.process_time,
                    node=self,
                    slot=slot
                )
            else:
                self.queue_to_enter.skip_process(process)

        self.queue_to_enter.recover()


    def maneuver_to_dispatch(self, simulator: DESSimulator):
        for slot in self.load_units + self.unload_units:
            if slot.current_train:
                print(f'{simulator.current_date}:: Train {slot.current_train.ID} entering on leaving queue!')
                train = slot.clear()
                simulator.add_event(
                    time=timedelta(),
                    callback=train.leave,
                    simulator=simulator,
                    node=self
                )
    # ====== Events ==========
    # ====== Methods ==========

    def __repr__(self):
        return self.name

    __str__ = __repr__

    def next_idle_slot(self, current_time) -> Slot:
        slots = sorted((slot for slot in self.slots), key=lambda slot: slot.time_to_be_idle(current_time))
        return slots[0]

    def time_to_call(self, current_time):
        process_scheduled_trains = len(self.train_schedule) * self.process_time
        # process_train_on_queue = self.queue_to_enter.current_size * self.process_time
        minimum_slot_time = self.next_idle_slot(current_time=current_time).time_to_be_idle(current_time=current_time)

        return minimum_slot_time + process_scheduled_trains

    def connect_neighbor(self, rail_segment: RailSegment):
        self.neighbors[rail_segment.destination.identifier] = rail_segment

    def predicted_time(self, current_time: datetime):
        return self.process_time + self.time_to_call(current_time)

