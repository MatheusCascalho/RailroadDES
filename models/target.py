from abc import ABC, abstractmethod
from models.demand import Demand


class TargetManager(ABC):
    @abstractmethod
    def furthest_from_the_target(self) -> Demand:
        pass


class SimpleTargetManager(TargetManager):
    def __init__(self, demand: list[Demand]):
        self.demand = demand
        self.target = {
            d.flow: {
                'demand':d,
                'target': d.volume
            } for d in demand
        }

    def furthest_from_the_target(self) -> Demand:
        rank = [
            {
                'demand': d['demand'],
                'distance_to_target': d['target'] - d['demand'].promised
            } for d in self.target.values()
        ]
        rank = sorted(rank, key=lambda x: x['distance_to_target'])
        return rank[0]['demand']
