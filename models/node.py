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
from petri_nets.petri_components import Place, Transition, Arc
from petri_nets.net import PetriNet
import numpy as np


@dataclass
class Neighbor:
    neighbor: NodeInterface
    transit_time: float


class Node(NodeInterface):
    def __init__(
            self,
            queue_capacity: int,
            name: Any,
            slots: int,
            process_time: timedelta,
            initial_trains: dict = {"receiving": 0, "processing": 0, "dispatching": 0}
    ):
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
        self._process_time = process_time
        self.initial_trains = initial_trains
        self.petri_model = self.build_petri_model()
        self.transitions = self.build_transitions_map()

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

    @property
    def processing_slots(self):
        return len([1 for slot in self.slots if slot.is_idle])

    # ====== Properties ==========
    # ====== Events ==========
    def call_to_enter(self, simulator: DESSimulator, train: TrainInterface, arrive: datetime):
        print(f"{simulator.current_date}:: Train {train.id} enter on queue of node {self}")
        self.queue_to_enter.push(
            element=train,
            arrive=arrive
        )
        time = self.time_to_call(current_time=simulator.current_date)
        self.train_schedule.append(train)
        self.state.trains_on_queue_to_enter = self.queue_to_enter.current_size

        # Add next event
        simulator.add_event(
            time=time,
            callback=self.process,
            simulator=simulator,
        )

    def process(self, simulator: DESSimulator):
        # Update resources
        slot = self.next_idle_slot(current_time=simulator.current_date)
        if slot.is_idle:
            print(
                f'{simulator.current_date}:: Train {self.queue_to_enter.elements[0].element.id} starts process at node {self}!')
            train = self.queue_to_enter.pop(
                current_time=simulator.current_date
            )
            self.train_schedule.pop(0)

            slot.put(
                train=train,
                date=simulator.current_date,
                time=self.process_time
            )

            # Update state
            self.state.trains_on_queue_to_enter = self.queue_to_enter.current_size
            self.state.trains_on_process = self.processing_slots
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
            time = slot.time_to_be_idle(current_time=simulator.current_date)
            simulator.add_event(
                time=time,
                callback=self.process,
                simulator=simulator,
            )
            print(
                f'{simulator.current_date}:: Train {self.queue_to_enter.elements[0].element.id} waiting idle slot at {self}!')

    def maneuver_to_dispatch(self, simulator: DESSimulator, slot: Slot):
        print(f'{simulator.current_date}:: Train {slot.current_train.id} entering on leaving queue!')
        train = slot.clear()
        self.queue_to_leave.push(
            element=train,
            arrive=simulator.current_date
        )

        simulator.add_event(
            time=timedelta(),
            callback=train.leave,
            simulator=simulator,
            node=self
        )
    # ====== Events ==========
    # ====== Methods ==========

    def build_petri_model(self):
        process_unit = Place(
            tokens=self.initial_trains.get('processing', 0),
            meaning="process unit",
            identifier=f"p_node_{self.name}_process_unit",
        )
        receiving_yard = Place(
            tokens=self.initial_trains.get('receiving', 0),
            meaning="yard to receiving trains",
            identifier=f"p_node_{self.name}_receiving_yard",
        )
        dispatch_yard = Place(
            tokens=self.initial_trains.get('dispatching', 0),
            meaning="yard to dispatch trains",
            identifier=f"p_node_{self.name}_dispatch_yard",
        )
        receive = Transition(
            intrinsic_time=timedelta(),
            meaning="receive trains on receiving yard",
            identifier=f"t_node_{self.name}_receive",
        )

        call = Transition(
            intrinsic_time=timedelta(),
            meaning="call train from receiving yard to process unit",
            identifier=f"t_node_{self.name}_call",
            callback=self.call_to_enter,
        )
        process = Transition(
            intrinsic_time=self.process_time,
            meaning="Execution of process",
            identifier=f"t_node_{self.name}_process",
            callback=self.process
        )
        dispatch = Transition(
            intrinsic_time=timedelta(),
            meaning="dispatch train to railroad",
            identifier=f"t_node_{self.name}_dispatch",
            callback=self.maneuver_to_dispatch
        )
        receiving_process = [
            Arc(input=receive, output=receiving_yard),
            Arc(input=receiving_yard, output=call)
        ]
        node_process = [
            Arc(input=call, output=process_unit),
            Arc(input=process_unit, output=process)
        ]
        dispatch_process = [
            Arc(input=process, output=dispatch_yard),
            Arc(input=dispatch_yard, output=dispatch)
        ]
        receiving_model = PetriNet(
            places=[receiving_yard],
            transitions=[receive, call],
            arcs=receiving_process,
            name=f"node_{self.name}_receiving_model",
            place_invariant=np.array([1]),
            token_restriction=self.queue_to_enter.capacity
        )
        process_model = PetriNet(
            places=[process_unit],
            transitions=[call, process],
            arcs=node_process,
            name=f"node_{self.name}_process_model",
            place_invariant=np.array([1]),
            token_restriction=len(self.slots)
        )
        dispatch_model = PetriNet(
            places=[dispatch_yard],
            transitions=[process, dispatch],
            arcs=dispatch_process,
            name=f"node_{self.name}_dispatch_model",
            place_invariant=np.array([1]),
            token_restriction=self.queue_to_leave.capacity
        )

        node_model = receiving_model.modular_composition([process_model, dispatch_model])

        return node_model

    def build_transitions_map(self):
        transition_map = {}
        for transition in self.petri_model.transitions:
            if "receive" in transition.identifier:
                transition_map['receive'] = transition
            elif "dispatch" in transition.identifier:
                transition_map['dispatch'] = transition
        return transition_map


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

    def connect_neighbor(self, node: NodeInterface, transit_time: float):
        self.neighbors[node.identifier] = Neighbor(
            neighbor=node,
            transit_time=transit_time
        )

    def predicted_time(self, current_time: datetime):
        return self.process_time + self.time_to_call(current_time)

