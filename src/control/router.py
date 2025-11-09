from collections import defaultdict
from dataclasses import asdict
from src.domain.entities.task import Task
from src.domain.entities.demand import Demand, Flow
from src.simulation.state_machine import ExpandableStateMachine, Transition, State
from src.domain.observers import AbstractObserver
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
        self.choosed_tasks = []
        self.completed_tasks = []
        self.running_tasks = {}

    def route(self, train: TrainInterface, current_time, state, is_initial=False):
        completed = train.current_task
        if completed.demand.flow.origin != 'origin':
            self.completed_tasks.append(completed)
        if completed in self.running_tasks:
            self.running_tasks.pop(completed)
        task = self.choose_task(
            current_time, 
            train_size=train.capacity, 
            model_state=state, 
            current_location=train.current_location,
        )
        self.choosed_tasks.append(task)
        train.current_task = task
        self.running_tasks[task] = train

    @abstractmethod
    def choose_task(self, current_time, train_size, model_state, **kwargs) -> Task:
        pass

    def operated_volume(self):
        return sum([d.operated for d in self.demands])

    def total_demand(self):
        return sum([d.volume for d in self.demands])
    
    @staticmethod
    def create_path(selected_demand: Demand, current_location: str):
        path = [selected_demand.flow.origin, selected_demand.flow.destination]
        is_moving = '_' in current_location or current_location == 'origin'
        if not is_moving:
            path.insert(0, current_location)
        return path, is_moving

    def demand_to_task(self, selected_demand, current_location, train_size, current_time, model_state):
        path, is_moving = self.create_path(selected_demand=selected_demand, current_location=current_location)
        task = Task(
            demand=selected_demand,
            path=path,
            task_volume=train_size,
            current_time=current_time,
            state=model_state,
            starts_moving=is_moving
        )
        return task
    
    def save(self, file: str):
        decisions = [asdict(t.demand.flow) for t in self.completed_tasks]
        with open(file, 'w') as f:
            json.dump(decisions, f, indent=2, ensure_ascii=False)

class RandomRouter(Router):
    def __init__(self, demands):
        super().__init__(demands=demands)

    def choose_task(self, current_time, train_size, model_state, **kwargs) -> Task:
        random_index = randint(0, len(self.demands)-1)
        random_demand = self.demands[random_index]
        task = self.demand_to_task(
            selected_demand=random_demand,
            current_location=kwargs.get('current_location', ''),
            train_size=train_size,
            current_time=current_time,
            model_state=model_state,
        )

        return task



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

    def route(self, train: TrainInterface, current_time: datetime, state: Any, current_location, *args, **kwargs) -> Task:
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
            task = self.demand_to_task(
                selected_demand=demand,
                current_location=current_location,
                train_size=kwargs.get('train_size'),
                current_time=current_time,
                model_state=kwargs.get('model_state'),
            )
        if not self.to_repeat and task is None:
            task = self.choose_task(current_time, train_size=train.capacity, model_state=state, current_location=current_location)

        # self.decision_map[task.model_state].append(task)
        train.current_task = task

    def choose_task(self, current_time, train_size, model_state, current_location) -> Task:
        random_index = randint(0, len(self.demands)-1)
        random_demand = self.demands[random_index]
        task = self.demand_to_task(
            selected_demand=random_demand,
            current_location=current_location,
            train_size=train_size,
            current_time=current_time,
            model_state=model_state,
        )

        return task