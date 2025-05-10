from dataclasses import dataclass, field, InitVar, asdict
from datetime import timedelta, datetime
from interfaces.node_interce import NodeInterface
from models.railroad import RailSegment
from collections import defaultdict
from typing import Union

@dataclass
class TransitTime:
    load_origin: str
    load_destination: str
    loaded_time: Union[float, timedelta]
    empty_time: Union[float, timedelta]

    def __post_init__(self):
        if isinstance(self.loaded_time, (float, int)):
            self.loaded_time = timedelta(hours=self.loaded_time)
        if isinstance(self.empty_time, (float, int)):
            self.empty_time = timedelta(hours=self.empty_time)

    def to_dict(self):
        data = asdict(self)
        data['loaded_time'] = data['loaded_time'].total_seconds() / (60 * 60)
        data['empty_time'] = data['empty_time'].total_seconds() / (60*60)
        return data



@dataclass
class RailroadMesh:
    load_points: tuple[NodeInterface]
    unload_points: tuple[NodeInterface]
    transit_times: list[TransitTime]
    name_to_node: dict[str, NodeInterface] = field(init=False, default_factory=dict)
    id_to_name: dict[int, str] = field(init=False, default_factory=dict)
    segments: list[RailSegment] = field(init=False, default_factory=list)
    graph: dict[NodeInterface, list[RailSegment]] = field(default_factory=lambda : defaultdict(list), init=False)


    def __post_init__(self):
        self.name_to_node: dict[str, NodeInterface] = {}
        self.id_to_name = {}
        for i, node in enumerate(self):
            self.name_to_node[node.name] = node
            self.id_to_name[node.identifier] = node.name

        for transit in self.transit_times:
            origin = self.name_to_node[transit.load_origin]
            destination = self.name_to_node[transit.load_destination]
            segment = RailSegment(
                origin=origin,
                destination=destination,
                time_to_origin=transit.loaded_time,
                time_to_destination=transit.empty_time
            )
            self.segments.append(segment)
            self.graph[origin.name].append(segment)
            self.graph[destination.name].append(segment.reversed())

    def __iter__(self):
        all_points = self.load_points + self.unload_points
        return all_points.__iter__()

    # def transit_time(self, origin_id, destination_id):
    #     is_loaded_transit = origin_id in [node.identifier for node in self.load_points]
    #     for transit in self.transit_times:
    #         if (
    #                 is_loaded_transit and
    #                 self.name_to_node[transit.load_origin].identifier == origin_id and
    #                 self.name_to_node[transit.load_destination].identifier == destination_id
    #         ):
    #             return transit.loaded_time
    #         elif (
    #                 not is_loaded_transit and
    #                 self.name_to_node[transit.load_origin].identifier == destination_id and
    #                 self.name_to_node[transit.load_destination].identifier == origin_id
    #         ):
    #             return transit.empty_time
    #     return timedelta(days=float('inf'))

    def node_by_id(self, identifier):
        name = self.id_to_name[identifier]
        node = self.name_to_node[name]
        return node

    # def predicted_time_for_path(self, path: list[int], current_time: datetime):
    #     total_time = timedelta()
    #     for origin_id, destination_id in zip(path[:-1], path[1:]):
    #         destination = self.node_by_id(destination_id)
    #         transit = self.transit_time(origin_id=origin_id, destination_id=destination_id)
    #         time_on_destination = destination.predicted_time(current_time=current_time)
    #         total_time += transit + time_on_destination
    #     return total_time

    # @staticmethod
    def complete_path(self, origin_name, destination_name):
        # Improvement: Dijkstra algorithm to complex mesh
        return [self.name_to_node[origin_name].identifier, self.name_to_node[destination_name].identifier]
