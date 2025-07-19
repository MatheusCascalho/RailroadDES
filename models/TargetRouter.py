from collections import defaultdict
from dataclasses import asdict
from models.task import Task
from models.demand import Demand, Flow
from models.state_machine import ExpandableStateMachine, Transition, State
from models.observers import AbstractObserver
from models.router import Router
from abc import ABC, abstractmethod
from random import randint
from interfaces.train_interface import TrainInterface
from datetime import datetime
from typing import Any
import json

class TargetManager(ABC):
    @abstractmethod
    def furthest_from_the_target(self):
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

    def furthest_from_the_target(self):
        rank = [
            {
                'demand': d['demand'],
                'distance_to_target': d['target'] - d['demand'].promised
            } for d in self.target.values()
        ]
        rank = sorted(rank, key=lambda x: x['distance_to_target'])
        return rank[0]['demand']

class TargetRouter(Router):
    def __init__(
            self,
            demands: list[Demand],
            target_manager_factory: callable = SimpleTargetManager
    ):
        super().__init__(demands=demands)
        self.target_manager: TargetManager = target_manager_factory(demands)


    def choose_task(self, current_time, train_size, model_state):
        furthest_demand = self.target_manager.furthest_from_the_target()
        task = Task(
            demand=furthest_demand,
            path=[furthest_demand.flow.origin, furthest_demand.flow.destination],
            task_volume=train_size,
            current_time=current_time,
            state=model_state
        )
        return task