from models.entities import Entity
import abc
from models.des_simulator import DESSimulator
from models.constants import (
    EPSILON,
    TrainActions
)
from dataclasses import dataclass, field, InitVar
from typing import Any, Generator, Callable
from datetime import timedelta, datetime
from models.exceptions import TrainExceptions
from models.states import (
    TrainState,
    TimeRegister,
    NodeState
)
from models.entities import Entity
from interfaces.node_interce import NodeInterface
from interfaces.train_interface import TrainInterface
from models.resources import Slot
import models.model_queue as mq


@dataclass
class Neighbor:
    neighbor: NodeInterface
    transit_time: float


class Node(NodeInterface):
    def __init__(self, queue_capacity: int, name: Any, slots: int):
        self._id = name
        self.name = name
        self.queue_to_enter = mq.Queue(capacity=queue_capacity)
        self.queue_to_leave = mq.Queue(capacity=float('inf'))
        self.slots: list[Slot] = [Slot() for _ in range(slots)]
        self.train_schedule: list[TrainInterface] = []
        self.state: NodeState = NodeState(
            average_time_on_queue_to_enter=timedelta(),
            average_time_on_queue_to_leave=timedelta(),

        )
        self.neighbors: dict[int, Neighbor] = {}

    # ====== Properties ==========
    @property
    def identifier(self):
        return self._id

    @identifier.setter
    def identifier(self, new_identifier: int):
        self._id = new_identifier

    @property
    def process_time(self) -> timedelta:
        return timedelta(hours=10.0)

    @property
    def processing_slots(self):
        return len([1 for slot in self.slots if slot.is_idle])

    # ====== Properties ==========
    # ====== Events ==========
    def call_to_enter(self, simulator: DESSimulator, train: TrainInterface, arrive: datetime):
        print(f"{simulator.current_date}:: Train enter on queue")
        time = self.time_to_call(current_time=simulator.current_date)
        self.queue_to_enter.push(
            element=train,
            arrive=arrive
        )
        self.state.trains_on_queue_to_enter = self.queue_to_enter.current_size

        # Add next event
        slot = self.next_idle_slot(current_time=simulator.current_date)
        simulator.add_event(
            time=time,
            callback=self.process,
            simulator=simulator,
            slot=slot
        )

    def process(self, simulator: DESSimulator, slot: Slot):
        # Update resources
        train = self.queue_to_enter.pop(
            current_time=simulator.current_date
        )
        slot.put(
            train=train,
            date=simulator.current_date,
            time=self.process_time
        )

        # Update state
        self.state.trains_on_queue_to_enter = self.queue_to_enter.current_size
        self.state.trains_on_process = self.processing_slots

        # Add next event
        simulator.add_event(
            time=timedelta(),
            callback=train.process,
            simulator=simulator,
            volume=5e3,
            start=simulator.current_date,
            process_time=self.process_time,
            node=self,
            slot=slot
        )

    def maneuver_to_dispatch(self, simulator: DESSimulator, slot: Slot):
        print(f'{simulator.current_date}:: Train entering on leaving queue!')
        train = slot.clear()
        # self.queue_to_leave.push(
        #     element=train,
        #     arrive=simulator.current_date
        # )

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
        slots = sorted(self.slots, key=lambda slot: slot.time_to_be_idle(current_time))
        return slots[0]

    def time_to_call(self, current_time):
        process_train_on_queue = self.queue_to_enter.current_size * self.process_time
        minimum_slot_time = self.next_idle_slot(current_time=current_time).time_to_be_idle(current_time=current_time)
        return process_train_on_queue + minimum_slot_time

    def connect_neighbor(self, node: NodeInterface, transit_time: float):
        self.neighbors[node.identifier] = Neighbor(
            neighbor=node,
            transit_time=transit_time
        )
