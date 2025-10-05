from pandas import Timedelta
from models import train
from models.router import Router
from models.demand import Demand, Flow
from models.railroad_mesh import RailroadMesh
from datetime import timedelta
from typing import Generator
from models.task import Task
from random import randint


class Solution:
    def __init__(self, flow_sequence: list[Flow]) -> None:
        self.flow_sequence = flow_sequence
        self.last_idx = 0

    def flows(self) -> Generator:
        for flow in self.flow_sequence:
            self.last_idx += 1
            yield flow

    def not_executed(self):
        return self.flow_sequence[self.last_idx:]
    
    def __repr__(self) -> str:
        return str([f.origin for f in self.flow_sequence])
    
    __str__ = __repr__


class SolutionBasedRouter(Router):
    def __init__(
            self, 
            demands: list[Demand], 
            train_size: float, 
            railroad_mesh: RailroadMesh,
            initial_solution: Solution | None = None
        ):
        super().__init__(demands=demands)
        self.train_size = train_size
        self.mesh = railroad_mesh
        self.demand_map = {
            d.flow: d
            for d in demands
        }
        self.__current_solution = initial_solution if initial_solution else self.basic_initial_solution()
        self.flow_sequence = self.__current_solution.flows()
        
    @property
    def solution(self):
        return self.__current_solution
    
    @solution.setter
    def solution(self, v):
        self.__current_solution = v
        self.flow_sequence = self.__current_solution.flows()

    def basic_initial_solution(self) -> Solution:
        flow_rank = [
            {"flow": d.flow, "demand": d.volume, "cycle_time": self.get_cycle_time(flow=d.flow)}
            for d in self.demands
        ]
        flow_sequence = []
        while flow_rank:
            flow_rank = sorted(flow_rank, key=lambda x: (-x['demand'], x["cycle_time"]))
            flow_sequence.append(flow_rank[0]['flow'])
            flow_rank[0]['demand'] -= self.train_size
            if flow_rank[0]['demand'] <= 0:
                flow_rank.pop(0)
        solution = Solution(flow_sequence=flow_sequence)
        return solution
    
    
    def get_cycle_time(self, flow: Flow) -> timedelta:
        segments = self.mesh.get_segments(path=[f"{flow.origin}-{flow.destination}"])
        cycle_time = timedelta()
        for i, segment in enumerate(segments):
            cycle_time += segment.time_to_destination + segment.time_to_origin
            if i == 0:
                cycle_time += segment.origin.process_time(train_size=6e3)
            if i == len(segments)-1:
                cycle_time += segment.destination.process_time(train_size=6e3)
        return cycle_time
    
    def choose_task(self, current_time, train_size, model_state, **kwargs) -> Task:
        for flow in self.flow_sequence:
            demand = self.demand_map[flow]
            task = self.demand_to_task(
                selected_demand=demand,
                current_location=kwargs.get('current_location', ''),
                train_size=train_size,
                current_time=current_time,
                model_state=model_state,
            )
            return task
        random_demand = self.demands[randint(0,len(self.demands)-1)]
        task = self.demand_to_task(
            selected_demand=random_demand,
            current_location=kwargs.get('current_location', ''),
            train_size=train_size,
            current_time=current_time,
            model_state=model_state,
        )
        return task
    
