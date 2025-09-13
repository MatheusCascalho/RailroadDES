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
from collections import Counter
from models.action_space import ActionSpace

EPSILON_DEFAULT = 1.0
N_NEURONS = 64
BATCH_SIZE = 50
GAMMA = 0.99
LEARNING_RATE = 1e-3
epsilon_min = 0.01
epsilon_decay = 0.89


class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_states, N_NEURONS),
            nn.ReLU(),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(N_NEURONS, n_actions),
        )

    def forward(self, x):
        return self.fc(x)


class Learner:
    def __init__(
            self,
            state_space: TFRStateSpace,
            action_space: ActionSpace,
            policy_net_path: str = 'serialized_models/policy_net.dill',
            target_net_path: str = 'serialized_models/target_net.dill',
            target_update_freq: int = 10_000,   # agora em steps, não episódios
            epsilon_start: float = .01,
            epsilon_end: float = 0.01,
            epsilon_decay_steps: int = 100,
    ):
        self._memory = deque(maxlen=100_000)
        self.state_space = state_space
        self.action_space = action_space

        suffix = f"{state_space.cardinality}x{self.action_space.n_actions}_TFRState_v2"
        offset = len('.dill')

        policy_net_path = policy_net_path[:-offset] + f"_{suffix}" + policy_net_path[-offset:]
        target_net_path = target_net_path[:-offset] + f"_{suffix}" + target_net_path[-offset:]

        self.policy_net = self.load_policies(policy_net_path)
        self.target_net = self.load_policies(target_net_path)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # sincronização inicial

        self.target_update_freq = target_update_freq
        self.policy_net_path = policy_net_path
        self.target_net_path = target_net_path

        # ε-greedy params
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start
        self.global_step = 0  # usado para decaimento linear

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
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

    def learn(self):
        # Amostra mini-batch
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor([self.state_space.to_array(s) for s in states])
        actions = torch.LongTensor([self.action_space.to_scalar(a) for a in actions]).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor([self.state_space.to_array(s) for s in next_states])
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # -------- Double DQN --------
        with torch.no_grad():
            # escolha da ação pela policy_net
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            # avaliação do valor da ação escolhida pela target_net
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + GAMMA * next_q * (1 - dones)

        # Q previsto
        current_q = self.policy_net(states).gather(1, actions)

        # Loss e otimização
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # estabilidade
        self.optimizer.step()

    def update(self, experience):
        self.memory.append(experience)
        self.global_step += 1

        # treina só se tiver memória suficiente
        if len(self.memory) >= BATCH_SIZE:
            self.learn()

        # atualização periódica da rede alvo
        if self.global_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # atualização do epsilon (linear)
        fraction = min(self.global_step / self.epsilon_decay_steps, 1.0)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

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
            learner: Learner,   # 🔹 novo: recebe o learner
            simulation_memory: RailroadEvolutionMemory,
            exploration_method: callable = None,
    ):
        super().__init__(demands=demands)
        self.action_space = ActionSpace(demands)
        self.completed_tasks = []
        self.running_tasks = {}
        self.state_space = state_space
        self.explore = exploration_method if exploration_method else self.action_space.sample
        self.policy_net = learner.policy_net
        self.memory = simulation_memory
        self.learner = learner   # 🔹 guardamos a referência

    def choose_task(self, current_time, train_size, model_state, current_location):
        roullete = random.random()
        if (
                self.memory.last_item is None or
                self.memory.last_item.state.is_initial or
                roullete < self.learner.epsilon    # 🔹 usa epsilon do Learner
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

        path = [selected_demand.flow.origin, selected_demand.flow.destination]
        is_moving = '_' in current_location or current_location == 'origin'
        if not is_moving:
            path.insert(0, current_location)

        task = Task(
            demand=selected_demand,
            path=path,
            task_volume=train_size,
            current_time=current_time,
            state=model_state,
            starts_moving=is_moving
        )

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
