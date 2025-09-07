from dataclasses import dataclass, field, InitVar, asdict
from datetime import timedelta, datetime
from interfaces.node_interce import NodeInterface
from collections import defaultdict
from typing import Union
from models.task import Task

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


class RailSegment:
    def __init__(
            self,
            origin: NodeInterface,
            destination: NodeInterface,
            time_to_origin: timedelta,
            time_to_destination: timedelta
    ):
        self.origin = origin
        self.destination = destination
        self.to_origin = []
        self.time_to_origin = time_to_origin
        self.to_destination = []
        self.time_to_destination = time_to_destination

    def reversed(self):
        segment = RailSegment(
            origin=self.destination,
            time_to_origin=self.time_to_destination,
            destination=self.origin,
            time_to_destination=self.time_to_origin
        )
        return segment

    def __repr__(self):
        return f"{self.origin} to {self.destination}"


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
        for origin in self.name_to_node:
            for destination in self.name_to_node:
                if origin == destination:
                    continue
                if (origin, destination) in [(s.origin, s.destination) for s in self.graph[origin]]:
                    continue

                segment = RailSegment(
                    origin=self.name_to_node[origin],
                    destination=self.name_to_node[destination],
                    time_to_origin=self.graph[origin][0].time_to_origin,
                    time_to_destination=self.graph[origin][0].time_to_destination
                )
                self.graph[origin].append(segment)
                if (destination, origin) in [(s.origin, s.destination) for s in self.graph[destination]]:
                    continue

                segment = RailSegment(
                    origin=self.name_to_node[destination],
                    destination=self.name_to_node[origin],
                    time_to_origin=self.graph[destination][0].time_to_origin,
                    time_to_destination=self.graph[destination][0].time_to_destination
                )
                self.graph[destination].append(segment)

    def __iter__(self):
        all_points = self.load_points + self.unload_points
        return all_points.__iter__()

    def node_by_id(self, identifier):
        name = self.id_to_name[identifier]
        node = self.name_to_node[name]
        return node

    def complete_path(self, origin_name, destination_name):
        # Improvement: Dijkstra algorithm to complex mesh
        return [self.name_to_node[origin_name].identifier, self.name_to_node[destination_name].identifier]

    def __len__(self):
        return len(self.load_points) + len(self.unload_points)

    def get_segments(self, task: Task):
        segments = []
        last = ''
        for n in task.path.path:
            if '-' not in n:
                continue
            o, d = n.split('-')
            if o in self.graph:
                s = self.graph[o][0]
            else:
                s = self.graph[d][0].reversed()
            segments.append(s)
        return segments

    def get_current_segment(self, task: Task) -> RailSegment:
        location = task.path.current_location
        o, d = location.split('-')
        if o == '_':
            possible_segments = [s for s in self.segments if s.destination.name == d]
            if not possible_segments:
                possible_segments = [s for s in self.segments if s.reversed().destination.name == d]
            s = possible_segments[0]
            return s
        if o in self.graph:
            for s in self.graph[o]:
                if s.destination.name == d:
                    return s
        raise Exception(f"No such segment: {location}")

    def to_json(self):
        return {
            'load_points': [node.to_json() for node in self.load_points],
            'unload_points': [node.to_json() for node in self.unload_points],
            'transit_times': [transit.to_dict() for transit in self.transit_times]
        }

