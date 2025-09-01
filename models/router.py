from collections import defaultdict
from dataclasses import asdict
from models.task import Task
from models.demand import Demand, Flow
from models.state_machine import ExpandableStateMachine, Transition, State
from models.observers import AbstractObserver
from abc import ABC, abstractmethod
from random import randint
from interfaces.train_interface import TrainInterface
from datetime import datetime
from typing import Any
import json


class Router(ABC):
    def __init__(self, demands: list[Demand]):
        self.demands = demands
        self.decision_map = defaultdict(list)
        self.completed_tasks = []
        self.running_tasks = {}

    def route(self, train: TrainInterface, current_time, state, is_initial=False):
        completed = train.current_task
        self.completed_tasks.append(completed)
        if completed in self.running_tasks:
            self.running_tasks.pop(completed)
        task = self.choose_task(
            current_time, 
            train_size=train.capacity, 
            model_state=state, 
            current_location=train.current_location,
        )
        self.decision_map[task.model_state].append(task)
        train.current_task = task
        self.running_tasks[task] = train

    @abstractmethod
    def choose_task(self, current_time, train_size, model_state):
        pass

    def operated_volume(self):
        return sum([d.operated for d in self.demands])

    def total_demand(self):
        return sum([d.volume for d in self.demands])

class RandomRouter(Router):
    def __init__(self, demands):
        super().__init__(demands=demands)

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

    def save(self, file: str):
        decisions = [asdict(t.demand.flow) for t in self.completed_tasks]
        with open(file, 'w') as f:
            json.dump(decisions, f, indent=2, ensure_ascii=False)

class TaskSpy(AbstractObserver):
    def __init__(self, recorder, phisical_state, valuable_state, previous_flow):
        super().__init__()
        self.recorder = recorder
        self.phisical_state = phisical_state
        self.valuable_state = valuable_state
        self.previous_flow = previous_flow

    def update(self):
        self.recorder.update(self)

    def get_task(self):
        return self.subjects[0]

class ChainedHistoryRouter(RandomRouter):
    def __init__(self, demands: list[Demand]):
        s = State('o', is_marked=True)
        s1 = State('d', is_marked=False)
        ghost = Transition('', s, s1)
        self.chained_decision_map = ExpandableStateMachine([ghost])
        super().__init__(demands)
        AbstractObserver().__init__()


    def route(self, train: TrainInterface, current_time, state, *args, **kwargs):
        previous_flow = train.current_task.demand.flow
        super().route(train, current_time, state)
        valuable_state = '\n'.join([str(d) for d in self.demands])
        spy = TaskSpy(self, phisical_state=state, valuable_state=valuable_state, previous_flow=previous_flow)
        train.current_task.add_observers(spy)

    def update(self, spy: TaskSpy):
        origin = (spy.phisical_state, spy.valuable_state, spy.previous_flow)
        destination = '\n'.join([str(d) for d in self.demands])
        trigger = spy.get_task()
        self.chained_decision_map.expand_machine(origin=origin, destination=destination, trigger=trigger)



class RepeatedRouter(Router):
    def __init__(self, demands, to_repeat: list[Flow]):
        super().__init__(demands=demands)
        self.demand_map = {d.flow: d for d in demands}
        self.to_repeat = to_repeat
        self.completed_tasks = []

    def route(self, train: TrainInterface, current_time: datetime, state: Any, *args, **kwargs) -> Task:
        completed = train.current_task
        self.completed_tasks.append(completed)
        task = None
        if self.to_repeat:
            not_found = True
            while not_found:
                if not self.to_repeat:
                    break
                choice = self.to_repeat.pop(0)
                demand = self.demand_map.get(choice)
                not_found = demand is None
            task = Task(
                demand=demand,
                path=[demand.flow.origin, demand.flow.destination],
                task_volume=train.capacity,
                current_time=current_time,
                state=state
            )
        if not self.to_repeat and task is None:
            task = self.choose_task(current_time, train_size=train.capacity, model_state=state)

        self.decision_map[task.model_state].append(task)
        train.current_task = task

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