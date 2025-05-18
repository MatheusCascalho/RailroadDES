from collections import defaultdict

from models.task import Task
from models.demand import Demand
from abc import ABC, abstractmethod
from random import randint
from interfaces.train_interface import TrainInterface
from datetime import datetime
from typing import Any


class Router(ABC):
    def __init__(self, demands: list[Demand]):
        self.demands = demands
        self.decision_map = defaultdict(list)

    @abstractmethod
    def route(self, train: TrainInterface, current_time: datetime, state: Any) -> Task:
        pass

class RandomRouter(Router):
    def __init__(self, demands):
        super().__init__(demands=demands)
        self.completed_tasks = []
        self.running_tasks = {}

    def route(self, train: TrainInterface, current_time, state):
        completed = train.current_task
        self.completed_tasks.append(completed)
        if completed in self.running_tasks:
            self.running_tasks.pop(completed)
        task = self.choose_task(current_time, train_size=train.capacity, model_state=state)
        self.decision_map[task.model_state].append(task)
        train.current_task = task
        self.running_tasks[task] = train

    def choose_task(self, current_time, train_size, model_state) -> Task:
        random_index = randint(0, len(self.demands)-1)
        random_demand = self.demands[random_index]
        task = Task(
            demand=random_demand,
            path=[random_demand.flow.origin, random_demand.flow.destination],
            task_volume=train_size,
            current_time=current_time,
            state=model_state
        )
        return task