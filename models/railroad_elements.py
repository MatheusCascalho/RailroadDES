from dataclasses import dataclass


def node_id_generator():
    identifier = 0
    while True:
        yield identifier
        identifier += 1


class Node:
    __id_generator = node_id_generator()

    def __init__(self, name: str, capacity: float, initial_trains: int = 0):
        self.name = name
        self.capacity = capacity
        self.identifier = next(self.__id_generator)
        self.initial_trains = initial_trains

    def __repr__(self):
        return f"{self.identifier}-{self.name}"

    __str__ = __repr__


@dataclass
class Flow:
    origin: Node
    destination: Node
    train_volume: float


@dataclass
class ExchangeBand:
    node: Node
    band: int


@dataclass
class TransitTime:
    origin: Node
    destination: Node
    time: float


@dataclass
class Demand:
    flow: Flow
    minimum: float
    maximum: float


class RailroadProblemTemplate:
    def __init__(
            self,
            trains: int,
            flows: list[Flow]
    ):
        self.__trains = trains
        self.__empty_origins = self.build_unload_points(flows=flows)
        self.__loaded_origins = self.build_load_points(flows=flows)
        self.__loaded_destinations = self.build_unload_points(flows=flows)
        u = len(self.__loaded_destinations)
        l = len(self.__loaded_origins)
        self.__cardinality = (self.__trains, u, l, u)

    @property
    def loaded_origins(self):
        return self.__loaded_origins

    @property
    def loaded_destinations(self):
        return self.__loaded_destinations

    @property
    def empty_origins(self):
        return self.__empty_origins

    @property
    def cardinality(self):
        return self.__cardinality

    @staticmethod
    def build_unload_points(flows: list[Flow]):
        nodes = set([f.destination for f in flows])
        nodes = sorted(nodes, key=lambda x: x.identifier)
        return nodes

    @staticmethod
    def build_load_points(flows: list[Flow]):
        nodes = set([f.origin for f in flows])
        nodes = sorted(nodes, key=lambda x: x.identifier)
        return nodes


