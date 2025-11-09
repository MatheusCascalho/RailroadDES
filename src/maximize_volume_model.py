"""
This file builds the Railroad Optimization Problem in the format to be used in Gurobi solver

"""
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

pd.set_option("display.max_columns", 10)


@dataclass
class RailroadResult:
    optimization_result: np.ndarray
    load_terminals: list[Node]
    unload_terminals: list[Node]
    demand: list[Demand]
    transit_times: np.ndarray

    def accpt_volume(self, verbose=True):
        accept = []
        train_volumes = np.zeros((len(self.load_terminals), len(self.unload_terminals)))
        travels = np.sum(self.optimization_result, axis=(0, 1))
        for demand in self.demand:
            train_volume = demand.flow.train_volume
            i = self.load_terminals.index(demand.flow.origin)
            j = self.unload_terminals.index(demand.flow.destination)
            train_volumes[i, j] = train_volume
            flow_travels = travels[i, j]
            flow_volume = flow_travels * train_volume
            if verbose:
                print(f"{demand.flow.origin}->{demand.flow.destination}: {flow_volume} TU \t")
            accept.append({
                "origin": demand.flow.origin,
                "destination": demand.flow.destination,
                "demand max": demand.maximum,
                "demand minimum": demand.minimum,
                "accept volume": flow_volume,
                "travels": flow_travels,
                "flow": f"{i}|{j}"
            })
        accept = pd.DataFrame(accept)
        return accept

    def empty_offer(self):
        report = []
        empty_travels = np.sum(self.optimization_result, axis=(0, 3))
        for i, origin in enumerate(self.unload_terminals):
            for j, destination in enumerate(self.load_terminals):
                travels = empty_travels[i, j]
                report.append({
                    "origin": origin,
                    "destination": destination,
                    "travels": travels
                })
        report = pd.DataFrame(report)
        return report

    def train_utilization(self, total_time):
        travel_times = self.optimization_result * self.transit_times
        time_by_train = np.sum(travel_times, axis=(1, 2, 3)) / total_time
        return time_by_train

    def travels_by_train(self):
        travels = np.sum(self.optimization_result, axis=(1, 2, 3))
        return travels




class RailroadOptimizationProblem:
    def __init__(
            self,
            trains: int,
            demands: list[Demand],
            transit_times: list[TransitTime],
            exchange_bands: list[ExchangeBand],
            time_horizon: int,
    ):
        # Building constraints
        self.demand = demands
        flows = [d.flow for d in demands]
        self.trains = trains
        self.capacity = CapacityRestrictions(trains=trains, flows=flows)
        print(f"Cardinality: {self.capacity.cardinality}")

        self.time_horizon = TimeHorizonRestriction(
            trains=trains,
            transit_times=transit_times,
            flows=flows,
            time_horizon=time_horizon
        )
        self.minimum_demand = MinimumDemandRestriction(trains=trains, demands=demands)
        self.maximum_demand = MaximumDemandRestriction(trains=trains, demands=demands)
        self.empty_offer = EmptyOfferRestriction(trains=trains, flows=flows)
        self.dispatch_initial_trains = DispatchInitialTrain(trains=trains, flows=flows)

        self.costs = sum([r.coefficients for r in self.maximum_demand.restrictions()])

        self.geq_constraints = []
        self.geq_constraints.extend(self.capacity.restrictions())
        self.geq_constraints.extend(self.time_horizon.restrictions())
        self.geq_constraints.extend(self.maximum_demand.restrictions())
        self.geq_constraints.extend(self.empty_offer.restrictions())

        self.leq_constraints = []
        self.leq_constraints.extend(self.minimum_demand.restrictions())
        self.leq_constraints.extend(self.dispatch_initial_trains.restrictions())

    def optimize(self, max_time):
        print(self)
        labels = self.labels()

        # Building Google OR-Tools model
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print('Solver not created!')
            return None

        # Building GUROBI model
        # Define variables (x)
        x = {}
        for i in labels:
            x[i] = solver.IntVar(0, solver.infinity(), f'x_{i}')

        # Define the objective function
        objective = solver.Objective()
        for i in x:
            objective.SetCoefficient(x[i], self.costs[i])
        objective.SetMaximization()

        # Add greater than or equal (geq) constraints
        for r in self.geq_constraints:
            constraint = solver.Constraint(-solver.infinity(), r.resource)
            for i in labels:
                constraint.SetCoefficient(x[i], r.coefficients[i])



        # Add less than or equal (leq) constraints
        for r in self.leq_constraints:
            constraint = solver.Constraint(r.resource, solver.infinity())
            for i in labels:
                constraint.SetCoefficient(x[i], r.coefficients[i])

        # Set time limit for solver
        solver.set_time_limit(max_time * 1000)  # Time limit in milliseconds

        # Solve the model
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            print("=" * 50)
            matrix = np.zeros(self.capacity.cardinality)
            for label in labels:
                matrix[label] = x[label].solution_value()

            result = RailroadResult(
                optimization_result=matrix,
                load_terminals=self.maximum_demand.loaded_origins,
                unload_terminals=self.maximum_demand.loaded_destinations,
                demand=self.demand,
                transit_times=self.time_horizon.transit_matrix
            )
            return result
        else:
            print('No solution found!')
            return None

    def labels(self):
        # Building vars label
        labels = []
        for n in range(self.trains):
            for i, e_origin in enumerate(self.capacity.empty_origins):
                for j, l_origin in enumerate(self.capacity.loaded_origins):
                    for k, l_destination in enumerate(self.capacity.loaded_destinations):
                        # label = f"train_{n}-empty_{i}-from_{j}_to-{k}"
                        label = (n, i, j, k)
                        labels.append(label)
        return labels

    def __repr__(self):
        repr = f"Problem with {len(self.geq_constraints+self.leq_constraints)} constraints and {len(self.labels())} variables\n\n"

        return repr

    def complete_repr(self):
        repr = f"Problem with {len(self.geq_constraints + self.leq_constraints)} constraints and {len(self.labels())} variables\n\n"
        for r in self.geq_constraints+self.leq_constraints:
            lhs = ""
            for i in self.labels():
                if r.coefficients[i] != 0:
                    lhs += str(r.coefficients[i]) + f"*x_{'|'.join([str(x) for x in i])} " + "\t\t"
            lhs = lhs[:-1].replace("\t\t", "\t+")
            repr += f"{lhs} {r.sense}= {r.resource} \n"

        return repr


if __name__=='__main__':
    n1 = Node(name='terminal 1', capacity=500e3)
    n2 = Node(name='terminal 2', capacity=500e3, initial_trains=1)
    n3 = Node(name='terminal 3', capacity=500e3, initial_trains=1)
    f1 = Flow(
        origin=n1,
        destination=n2,
        train_volume=5e3
    )
    f2 = Flow(
        origin=n1,
        destination=n3,
        train_volume=6e3
    )

    demands = [
        Demand(flow=f1, minimum=5e3, maximum=50e3),
        Demand(flow=f2, minimum=4e3, maximum=40e3),
    ]

    transit_times = [
        TransitTime(origin=n1, destination=n2, time=2.5),
        TransitTime(origin=n1, destination=n3, time=3.9),
    ]
    problem = RailroadOptimizationProblem(
        trains=2,
        transit_times=transit_times,
        demands=demands,
        exchange_bands=[],
        time_horizon=30
    )
    print(problem.complete_repr())
    result = problem.optimize(max_time=60)
    print(result.accpt_volume(verbose=False))

    print("="*50)
    print("Relatório de envio de vazios")
    print(result.empty_offer())
    print("="*50)
    print("Utilização de trem")
    print(result.train_utilization(total_time=30))