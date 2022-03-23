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


class Node(NodeInterface):
    from models.train import Train
    import models.model_queue as mq
    from models.resources import Slot

    def __init__(self, queue_capacity: int, identifier: int, slots: int):
        self._id = identifier
        self.queue_to_enter = mq.Queue(capacity=queue_capacity)
        self.queue_to_leave = mq.Queue(capacity=float('inf'))
        self.slots: list[Slot] = [Slot() for _ in range(slots)]
        self.train_schedule: list[Train] = []
        self.state: NodeState = NodeState(
            average_time_on_queue_to_enter=timedelta(),
            average_time_on_queue_to_leave=timedelta(),

        )

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

    def next_idle_slot(self, current_time) -> Slot:
        slots = sorted(self.slots, key=lambda slot: slot.time_to_be_idle(current_time))
        return slots[0]


    # ====== Properties ==========
    # ====== Events ==========
    def call_to_enter(self, simulator: DESSimulator, train: Train, arrive):
        self.queue_to_enter.push(
            element=train,
            arrive=arrive
        )
        self.state.trains_on_queue_to_enter = self.queue_to_enter.current_size

        # Add next event
        process_current_train = self.next_idle_slot(simulator.time).time_to_be_idle(simulator.time)
        process_trains_in_queue = self.queue_to_enter.current_size * self.process_time
        time = simulator.time + process_current_train + process_trains_in_queue
        simulator.add_event(
            time=time,
            callback=self.process,
            simulator=simulator
        )

    def process(self, simulator: DESSimulator, slot: Slot):
        # Update resources
        train = self.queue_to_enter.pop(
            current_time=simulator.time
        )
        slot.put(
            train=train,
            date=simulator.time,
            time=self.process_time
        )

        # Update state
        self.state.trains_on_queue_to_enter = self.queue_to_enter.current_size
        self.state.trains_on_process = self.processing_slots

        # Add next event
        simulator.add_event(
            time=simulator.time,
            callback=train.next_process,
            volume=5e3,
            start=simulator.time,
            process_time=self.process_time
        )

    def maneuver_to_dispatch(self, simulator, train):
        ...
