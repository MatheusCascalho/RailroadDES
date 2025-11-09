from src.target import TargetManager
from src.maximize_volume_model import RailroadOptimizationProblem
import numpy as np
from src.railroad_elements import Flow, TransitTime, Demand, ExchangeBand, Node
from src.restrictions import Restrictions, RestrictionType, Restriction
from src.capacity_restriction import CapacityRestrictions
from src.time_horizon_restriction import TimeHorizonRestriction
from src.empty_offer_restriction import EmptyOfferRestriction
from src.dispatch_initial_train_restriction import DispatchInitialTrain
from src.demand_restriction import MaximumDemandRestriction, MinimumDemandRestriction
from dataclasses import dataclass
import pandas as pd
from ortools.linear_solver import pywraplp
from src.railroad import Railroad

class MaximizeVolumeTarget(TargetManager):
    def __init__(self, model: Railroad):
        self.des_model = model
        self.demands = {
            d.flow: d
            for d in self.des_model.demands
        }
        self.problem = self.build_optimization_model()
        self.solution = self.problem.optimize(max_time=60)
        result = self.solution.accpt_volume(verbose=True)
        self.target = {}
        for i, row in result.iterrows():
            o = row['origin'].name
            d = row['destination'].name
            f = [f for f in self.demands
                 if f.origin == o and f.destination == d][0]
            target = row['accept volume']
            self.target[f] = {
                'demand': self.demands[f],
                'target': target
            }

        # self.target = {
        #     d.flow: {
        #         'demand': d,
        #         'target': d.volume
        #     } for d in demand
        # }

    def furthest_from_the_target(self) -> Demand:
        rank = [
            {
                'demand': d['demand'],
                'distance_to_target': d['target'] - d['demand'].promised
            } for d in self.target.values()
        ]
        rank = sorted(rank, key=lambda x: x['distance_to_target'], reverse=True)
        return rank[0]['demand']


    def build_optimization_model(self):
        problem = RailroadOptimizationProblem(
            trains=len(self.des_model.trains),
            transit_times=[
                TransitTime(
                    origin=Node(
                        name=tt.load_origin,
                        capacity=self.des_model.mesh.name_to_node[tt.load_origin].daily_capacity * 30
                    ),
                    destination=Node(
                        name=tt.load_destination,
                        capacity=self.des_model.mesh.name_to_node[tt.load_destination].daily_capacity * 30
                    ),
                    time=tt.loaded_time
                )
                for tt in
                self.des_model.mesh.transit_times
            ],
            demands=[
                Demand(
                    flow=Flow(
                        origin=Node(
                            name=d.flow.origin,
                            capacity=self.des_model.mesh.name_to_node[d.flow.origin].daily_capacity * 30
                        ),
                        destination=Node(
                            name=d.flow.destination,
                            capacity=self.des_model.mesh.name_to_node[d.flow.destination].daily_capacity * 30
                        ),
                        train_volume=6e3
                    ),
                    minimum=0,
                    maximum=d.volume)
                for d in self.des_model.demands
            ],
            exchange_bands=[],
            time_horizon=30
        )
        return problem

    def furthest_from_the_target(self) -> Demand:
        pass

if __name__ == "__main__":
    import dill
    # n1 = Node(name='terminal 1', capacity=500e3)
    # n2 = Node(name='terminal 2', capacity=500e3, initial_trains=1)
    # n3 = Node(name='terminal 3', capacity=500e3, initial_trains=1)
    # f1 = Flow(
    #     origin=n1,
    #     destination=n2,
    #     train_volume=5e3
    # )
    # f2 = Flow(
    #     origin=n1,
    #     destination=n3,
    #     train_volume=6e3
    # )
    #
    # demands = [
    #     Demand(flow=f1, minimum=5e3, maximum=50e3),
    #     Demand(flow=f2, minimum=4e3, maximum=40e3),
    # ]
    #
    # transit_times = [
    #     TransitTime(origin=n1, destination=n2, time=2.5),
    #     TransitTime(origin=n1, destination=n3, time=3.9),
    # ]
    # problem = RailroadOptimizationProblem(
    #     trains=2,
    #     transit_times=transit_times,
    #     demands=demands,
    #     exchange_bands=[],
    #     time_horizon=30
    # )
    # print(problem.complete_repr())
    # result = problem.optimize(max_time=60)

    with open('../tests/artifacts/model_to_train.dill', 'rb') as f:
        model = dill.load(f)

    optimal_target = MaximizeVolumeTarget(model)
    result = optimal_target.solution
    print(result.accpt_volume(verbose=False))

    print("="*50)
    print("Relatório de envio de vazios")
    print(result.empty_offer())
    print("="*50)
    print("Utilização de trem")
    print(result.train_utilization(total_time=30))