from collections import deque

from interfaces.train_interface import TrainInterface
from models.TFRState import TFRStateSpace
from models.observers import AbstractObserver
from models.router import Router
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

EPSILON_DEFAULT = 1.0
N_NEURONS = 64
BATCH_SIZE = 50
GAMMA = 0.99
LEARNING_RATE = 1e-3
epsilon_min = 0.01
epsilon_decay = 0.995


class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_states, N_NEURONS),
            nn.ReLU(),
            nn.Linear(N_NEURONS, n_actions),
        )

    def forward(self, x):
        return self.fc(x)

class ActionSpace:
    def __init__(self, demands):
        self.demands = demands + ['AUTOMATIC']

    @property
    def n_actions(self):
        return len(self.demands)

    def sample(self):
        i = random.randint(0, len(self.demands) - 2)
        return self.demands[i]

    def to_scalar(self, action):
        flows = [str(d.flow) for d in self.demands[:-1]] + [self.demands[-1]]
        v = flows.index(action)
        return v

    def get_demand(self, i):
        if i == len(self.demands) - 1:
            return self.sample()
        return self.demands[i]

class Learner:
    def __init__(
            self,
            state_space: TFRStateSpace,
            action_space: ActionSpace,
            policy_net_path: str = '../serialized_models/policy_net.dill',
            target_net_path: str = '../serialized_models/target_net.dill',
            target_update_freq: int = 10,
            epsilon: float = EPSILON_DEFAULT,
    ):
        self._memory = deque(maxlen=100_000)
        self.state_space = state_space
        self.action_space = action_space
        suffix = f"{state_space.cardinality}x{self.action_space.n_actions}_TFRState_v1"
        offset = len('.dill')

        policy_net_path = policy_net_path[:-offset] + f"_{suffix}" + policy_net_path[-offset:]
        target_net_path = target_net_path[:-offset] + f"_{suffix}" + target_net_path[-offset:]

        self.policy_net = self.load_policies(policy_net_path)
        self.target_net = self.load_policies(target_net_path)

        self.target_update_freq = target_update_freq
        self.policy_net_path = policy_net_path
        self.target_net_path = target_net_path
        self.epsilon = epsilon

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.episode = 0
        super().__init__()

    def load_policies(self, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                policy = dill.load(f)
        except FileNotFoundError:
            policy = DQN(
                n_states=self.state_space.cardinality,
                n_actions=self.action_space.n_actions,
            )
        return policy

    @property
    def memory(self):
        return self._memory

    @property
    def memory_to_train(self):
        return [e for e in self._memory if e.action != 'AUTOMATIC']

    def learn(self):
        # Amostra mini-batch
        batch = random.sample(self.memory_to_train, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = [self.state_space.to_array(s) for s in states]
        states = torch.FloatTensor(states)
        actions = [self.action_space.to_scalar(a) for a in actions]
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = [self.state_space.to_array(s) for s in next_states]
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Q-alvo
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + GAMMA * max_next_q * (1 - dones)

        # Q previsto
        current_q = self.policy_net(states).gather(1, actions)

        # Loss e otimização
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update(self, experience):
        self.memory.append(experience)
        self.episode += 1
        if len(self.memory_to_train) >= BATCH_SIZE:
            self.learn()

        if self.episode % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def __enter__(self, *args, **kwargs):
        debug(f"Início da aprendizagem do DQNRouter")
        return self

    def __exit__(self, *args, **kwargs):
        self.save(*args, **kwargs)

    def save(self, *args, **kwargs):
        with open(self.policy_net_path, 'wb') as f:
            dill.dump(self.policy_net, f)

        with open(self.target_net_path, 'wb') as f:
            dill.dump(self.target_net, f)


class DQNRouter(Router):
    def __init__(
            self,
            demands,
            state_space: TFRStateSpace,
            policy_net: DQN,
            simulation_memory: RailroadEvolutionMemory,
            exploration_method: callable = None,
            epsilon: float = EPSILON_DEFAULT,
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

    def choose_task(self, current_time, train_size, model_state):
        if self.memory.last_item is None or random.random() < self.epsilon:
            selected_demand = self.explore()
        else:
            with torch.no_grad():
                state = self.memory.last_item.next_state
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

    def update_epsilon(self):
        # Decaimento do epsilon
        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay
