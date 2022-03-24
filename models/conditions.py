import abc
from models.des_simulator import DESSimulator
import models.model_queue as mq
from dataclasses import dataclass, field, InitVar
from typing import Any, Generator, Callable
from datetime import timedelta, datetime
from interfaces.node_interce import NodeInterface


@dataclass
class TransitTime:
    load_origin: str
    load_destination: str
    loaded_time: timedelta
    empty_time: timedelta


@dataclass
class RailroadMesh:
    load_points: tuple[NodeInterface]
    unload_points: tuple[NodeInterface]
    transit_times: list[TransitTime]
    name_to_node: dict[str, NodeInterface] = field(init=False, default_factory=dict)
    id_to_name: dict[int, str] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.name_to_node: dict[str, NodeInterface] = {}
        self.id_to_name = {}
        for i, node in enumerate(self):
            node.identifier = i
            self.name_to_node[node.name] = node
            self.id_to_name[node.identifier] = node.name

        for transit in self.transit_times:
            origin = self.name_to_node[transit.load_origin]
            destination = self.name_to_node[transit.load_destination]
            origin.connect_neighbor(node=destination, transit_time=transit.loaded_time)
            destination.connect_neighbor(node=origin, transit_time=transit.empty_time)

    def __iter__(self):
        all_points = self.load_points + self.unload_points
        return all_points.__iter__()

    def transit_time(self, origin_id, destination_id):
        is_loaded_transit = origin_id in [node.identifier for node in self.load_points]
        for transit in self.transit_times:
            if (
                    is_loaded_transit and
                    self.name_to_id[transit.load_origin] == origin_id and
                    self.name_to_id[transit.load_destination] == destination_id
            ):
                return transit.loaded_time
            elif not is_loaded_transit and transit.load_origin == destination_id and transit.load_destination == origin_id:
                return transit.empty_time
        return timedelta(days=float('inf'))

    def node_by_id(self, identifier):
        name = self.id_to_name[identifier]['name']
        node = self.name_to_id[name]['node']
        return node

