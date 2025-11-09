from dataclasses import dataclass

from interfaces.train_interface import TrainInterface
from src.node_data_model import NodeData
from src.demand import Flow, Demand
from src.railroad_mesh import TransitTime


@dataclass
class MeshData:
    load_points: list[NodeData]
    unload_points: list[NodeData]
    transit_times: list[TransitTime]


@dataclass
class RailroadData:
    mesh: MeshData
    trains: int
    demands: list[Demand]

    def __post_init__(self):
        self.mesh = MeshData(**self.mesh)
        self.demands = [Demand(**d) for d in self.demands]