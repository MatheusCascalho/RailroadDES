import os
from collections import deque

from interfaces.train_interface import TrainInterface
from models.TFRState import TFRStateSpace
from models.observers import AbstractObserver
from models.router import Router
from models.states import ActivityState
from models.system_evolution_memory import RailroadEvolutionMemory
from datetime import datetime
from typing import Any
from models.task import Task
import torch.nn as nn
import torch.optim as optim
import torch
import dill
import random
import numpy as np
from logging import debug
from collections import Counter, defaultdict
from models.TFRState import TFRState
from models.action_space import ActionSpace
from models.system_evolution_memory import Experience, RailroadEvolutionMemory

ALPHA = 0.2
GAMMA = 0.9


class QTable(AbstractObserver):
    def __init__(self,  action_space: ActionSpace, learning_rate=ALPHA, discount_factor=GAMMA):
        self.q_table = defaultdict(lambda : defaultdict(float))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.action_space = action_space
        super().__init__()

    def learn(self, current_state, next_state: TFRState, action):
        q_actual = self.q_table.get(current_state, 0).get(action)
        reward = next_state.reward()
        max_future_value = max(self.q_table[next_state, a] for a in self.action_space[next_state])
        q_next = q_actual + self.learning_rate * (reward + self.dicount_factor * max_future_value - q_actual)
        self.q_table[current_state, action] = q_next

    def update(self):
        current_state = self.memory.last_item.state
        next_state = self.memory.last_item.next_state
        self.learn(
            current_state=current_state,
            next_state=next_state,
            action=self.memory[-1].action
        )

    @property
    def memory(self) -> RailroadEvolutionMemory:
        return self.subjects[0]


class QRouter(Router):
    def __init__(
            self,
            demands,
            state_space: TFRStateSpace,
            simulation_memory: RailroadEvolutionMemory,
            exploration_method: callable = None,
    ):
        super().__init__(demands=demands)
        self.action_space = ActionSpace(demands)
        self.completed_tasks = []
        self.running_tasks = {}
        self.state_space = state_space
        self.explore = exploration_method if exploration_method else self.action_space.sample
        self.policy_net = policy_net
        self.memory = simulation_memory
        self.epsilon = epsilon
        self.epsilon_steps = 0

    def choose_task(self, current_time, train_size, model_state):
        if (
                self.memory.last_item is None or
                self.memory.last_item.state.is_initial or
                random.random() < self.epsilon
        ):
            selected_demand = self.explore()
        else:
            with torch.no_grad():
                state = self.memory.last_item.next_state
                train_activities = Counter(t.activity for t in state.train_states)
                if train_activities.get(ActivityState.WAITING_TO_ROUTE) != 1:
                    raise Exception('It is not possible to identify the train that will be routed by the system state')
                state = self.state_space.to_array(state)
                state = torch.FloatTensor(state).unsqueeze(0)
                demand_index = self.policy_net(state).argmax().item()
                selected_demand = self.action_space.get_demand(demand_index)
        task = Task(
            demand=selected_demand,
            path=[selected_demand.flow.origin, selected_demand.flow.destination],
            task_volume=train_size,
            current_time=current_time,
            state=model_state
        )
        self.update_epsilon()
        return task

    def route(self, train: TrainInterface, current_time, state, is_initial=False):
        if is_initial:
            self.memory.save_previous_state(is_initial=True)
            super().route(train=train, current_time=current_time, state=state, is_initial=is_initial)
            self.memory.save_consequence()
        else:
            super().route(train=train, current_time=current_time, state=state, is_initial=is_initial)


    def update_epsilon(self):
        # Decaimento do epsilon
        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay
            self.epsilon_steps += 1
