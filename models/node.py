from models.exceptions import ProcessException
from models.node_constraints import ProcessConstraintSystem
from models.processors import ProcessorSystem, ProcessorRate
from models.des_simulator import DESSimulator
from models.constants import (
    Process
)
from dataclasses import dataclass
from typing import Any
from datetime import timedelta, datetime
from models.states import (
    ProcessorState
)
from models.railroad import RailSegment
from interfaces.node_interce import NodeInterface
import models.model_queue as mq
from models.clock import Clock
from models.stock_replanish import StockReplenisherInterface
from models.stock import StockInterface


@dataclass
class Neighbor:
    neighbor: NodeInterface
    transit_time: float


class Node(NodeInterface):
    def __init__(
            self,
            queue_capacity: int,
            name: Any,
            clock: Clock,
            process_rates: dict[str, list[ProcessorRate]],
            load_constraint_system: ProcessConstraintSystem,
            unload_constraint_system: ProcessConstraintSystem,
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
        self.load_units: list[ProcessorSystem] = self.build_load_units(process_rates)
        self.unload_units: list[ProcessorSystem] = self.build_unload_units(process_rates)
        self.neighbors: dict[int, RailSegment] = {}

    def build_load_units(self, process_rates: dict[str, list[ProcessorRate]]) -> list[ProcessorSystem]:
        load_units = [
            ProcessorSystem(
                processor_type=Process.LOAD,
                queue_to_leave=self.queue_to_leave,
                clock=self.clock,
                rates={p: rate}
            )
            for p, rates in process_rates.items()
            for rate in rates
            if rate.type == Process.LOAD
        ]
        for unit in load_units:
            unit.state_machine.states[ProcessorState.BUSY].add_observers(
                self.process_constraint[Process.LOAD].state_machine.transitions["start"]
            )
            unit.state_machine.states[ProcessorState.IDLE].add_observers(
                self.process_constraint[Process.LOAD].state_machine.transitions["finish"]
            )
        return load_units

    def build_unload_units(self, process_rates: dict[str, list[ProcessorRate]]) -> list[ProcessorSystem]:
        unload_units = [
            ProcessorSystem(
                processor_type=Process.UNLOAD,
                queue_to_leave=self.queue_to_leave,
                clock=self.clock,
                rates={p: rate}
            )
            for p, rates in process_rates.items()
            for rate in rates
            if rate.type == Process.UNLOAD
        ]
        for unit in unload_units:
            unit.state_machine.states[ProcessorState.BUSY].add_observer(
                self.process_constraint[Process.UNLOAD].state_machine.transitions["start"]
            )
            unit.state_machine.states[ProcessorState.IDLE].add_observer(
                self.process_constraint[Process.UNLOAD].state_machine.transitions["finish"]
            )
        return unload_units

    # ====== Events ==========
    def receive(self, train):
        self.queue_to_enter.push(train, arrive=self.clock.current_time)

    def dispatch(self):
        while self.queue_to_leave.is_busy:
            train = self.queue_to_leave.pop(self.clock.current_time)
            next_node = train.next_location
            self.neighbors[next_node].send(train)

    def process(self, simulator: DESSimulator):
        self.pre_processing()
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
                    process_time=slot.get_process_time(),
                    node=self,
                    slot=slot
                )
            else:
                self.queue_to_enter.skip_process(process)

        self.queue_to_enter.recover()
        self.pos_processing()

    def pos_processing(self):
        pass

    def pre_processing(self):
        pass

    def maneuver_to_dispatch(self, simulator: DESSimulator):
        self.pre_processing()
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
        self.pos_processing()
    # ====== Events ==========
    # ====== Methods ==========

    def __repr__(self):
        return self.name

    __str__ = __repr__

    def time_to_call(self, current_time):
        process_scheduled_trains = len(self.train_schedule) * self.process_time
        # process_train_on_queue = self.queue_to_enter.current_size * self.process_time
        minimum_slot_time = self.next_idle_slot(current_time=current_time).time_to_be_idle(current_time=current_time)

        return minimum_slot_time + process_scheduled_trains

    def connect_neighbor(self, rail_segment: RailSegment):
        self.neighbors[rail_segment.destination.identifier] = rail_segment

    def predicted_time(self, current_time: datetime):
        return self.process_time + self.time_to_call(current_time)
    # ====== Methods ==========
    # ====== Properties ==========
    @property
    def identifier(self):
        return self._id

    @identifier.setter
    def identifier(self, new_identifier: int):
        self._id = new_identifier

    @property
    def process_time(self) -> timedelta:
        return self._process_time

    # ====== Properties ==========
