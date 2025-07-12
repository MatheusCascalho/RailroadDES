from interfaces.train_interface import TrainInterface
from models.TFRState import TFRStateSpace
from models.router import Router
from datetime import datetime
from typing import Any
from models.task import Task
import torch.nn as nn
import torch.optim as optim
import dill


class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.fc(x)

class DQNRouter(Router):
    def __init__(
            self,
            demands,
            state_space: TFRStateSpace = None,
            policy_net_path: str = '../serialized_models/policy_net.dill',
            target_net_path: str = '../serialized_models/target_net.dill',
    ):
        super().__init__(demands=demands)
        self.completed_tasks = []
        self.running_tasks = {}
        self.state_space = state_space
        self.policy_net_path = policy_net_path
        self.target_net_path = target_net_path
        self.policy_net = self.load_policies(policy_net_path)
        self.target_net = self.load_policies(target_net_path)

    def load_policies(self, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                policy = dill.load(f)
        except FileNotFoundError:
            n_actions = len({d.flow for d in self.demands})
            policy = DQN(
                n_states=self.state_space.cardinality,
                n_actions=n_actions
            )
        return policy


    def route(self, train: TrainInterface, current_time: datetime, state: Any) -> Task:
        ...

    def __del__(self):
        with open(self.policy_net_path, 'wb') as f:
            dill.dump(self.policy_net, f)

        with open(self.target_net_path, 'wb') as f:
            dill.dump(self.target_net, f)
