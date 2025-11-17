import os
from collections import deque

from interfaces.train_interface import TrainInterface
from src.control.memory_based_control.reinforcement_learning.TFRState import TFRStateSpace
from src.standards.observers import AbstractObserver
from src.control.router import Router
from src.domain.states import ActivityState
from src.control.memory_based_control.system_evolution_memory import RailroadEvolutionMemory
from datetime import datetime
from typing import Any
from src.domain.entities.task import Task
import torch.nn as nn
import torch.optim as optim
import torch
import dill
import random
import numpy as np
from logging import debug
from collections import Counter
from src.control.memory_based_control.reinforcement_learning.action_space import ActionSpace
from typing import Callable
import logging
import numpy as np


EPSILON_DEFAULT = 1.0
N_NEURONS = 256
BATCH_SIZE = 100
GAMMA = 0.9
LEARNING_RATE = 1e-4
epsilon_min = 0.01
epsilon_decay = 0.89


class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_states, N_NEURONS),
            nn.ReLU(),
            nn.Linear(N_NEURONS, N_NEURONS),
            nn.ReLU(),
            nn.Linear(N_NEURONS, N_NEURONS),
            nn.ReLU(),
            nn.Linear(N_NEURONS, n_actions),
        )

    def forward(self, x):
        q_values = self.fc(x)
        return q_values

def learner_id_gen():
    i = 0
    while True:
        yield f"Learner_{i}"

leaner_id = learner_id_gen()

class Learner(AbstractObserver):
    def __init__(
            self,
            state_space: TFRStateSpace,
            action_space: ActionSpace,
            policy_net_path: str = 'serialized_models/policy_net.pytorch',
            target_net_path: str = 'serialized_models/target_net.pytorch',
            target_update_freq: int = 2_000,   # agora em steps, não episódios
            epsilon_start: float = 1,
            epsilon_end: float = 0.001,
            epsilon_decay_steps: int = 100,
            batch_size: int = BATCH_SIZE
    ):
        self.batch_size = batch_size
        self._memory = deque(maxlen=100_000)
        self.state_space = state_space
        self.action_space = action_space

        suffix = f"{state_space.cardinality}x{self.action_space.n_actions}_TFRState_v2"
        offset = len('.pytorch')

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

        # Configuração de logger
        l_id = next(leaner_id)
        self.logger = logging.getLogger(l_id)
        self.logger.setLevel(logging.INFO)
        # evita múltiplos handlers duplicados
        if not self.logger.handlers:
            log_dir = "logs/learning"
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(log_dir, f"{l_id}.log"))
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            sh = logging.StreamHandler()
            self.logger.addHandler(sh)

        self.logger.info(f"Learner inicializado com Double DQN + ε linear{'' if self.batch_size == BATCH_SIZE else 'TREINAMENTO ONLINE'}")

        # ---- buffers de métricas ----
        self.losses = deque(maxlen=500)
        self.rewards = deque(maxlen=500)
        self.q_values = deque(maxlen=500)

    def load_policies(self, filepath: str):
        model = DQN(
            n_states=self.state_space.cardinality,
            n_actions=self.action_space.n_actions,
        )
        if os.path.exists(filepath):
            try:
                model.load_state_dict(torch.load(filepath, map_location="cpu"))
                print(f"[INFO] Pesos carregados de {filepath}")
            except Exception as e:
                print(f"[WARN] Falha ao carregar {filepath}: {e}")
                print("[INFO] Nova rede DQN criada.")
        else:
            print(f"[INFO] Nenhum modelo encontrado em {filepath}. Nova rede criada.")
        return model

    @property
    def memory(self):
        return self._memory
    
    @property
    def memory_to_train(self):
        return self.memory if self.batch_size == BATCH_SIZE else [e for e in self.memory if e.action not in ['AUTOMATIC', 'ROUTING']]

    def learn(self):
        # Amostra mini-batch
        batch = random.sample(self.memory_to_train, self.batch_size)
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

        # ---- logging interno ----
        self.losses.append(loss.item())
        self.rewards.extend(rewards.squeeze().tolist())
        self.q_values.extend(current_q.detach().squeeze().tolist())
        freq = 100 if self.batch_size == BATCH_SIZE else 10
        if self.global_step % freq == 0:  # log periódico
            avg_loss = np.mean(list(self.losses)) if self.losses else 0
            avg_reward = np.mean(list(self.rewards)) if self.rewards else 0
            avg_q = np.mean(list(self.q_values)) if self.q_values else 0
            self.logger.info(
                f"[step={self.global_step}] "
                f"ε={self.epsilon:.3f} | "
                f"loss={avg_loss:.6f} | "
                f"reward_avg={avg_reward:.6f} | "
                f"Q_avg={avg_q:.6f}"
            )

    def update(self, experience=None):
        if experience is None:
            experience = self.subjects[0].last_item
        self.memory.append(experience)
        self.global_step += 1

        # treina só se tiver memória suficiente
        if len(self.memory_to_train) >= self.batch_size:
            self.learn()

        # atualização periódica da rede alvo
        if self.global_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.logger.info('Target net Updated!')

        # atualização do epsilon (linear)
        fraction = min(self.global_step / self.epsilon_decay_steps, 1.0)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def save(self, *args, **kwargs):
        os.makedirs(os.path.dirname(self.policy_net_path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), self.policy_net_path)
        torch.save(self.target_net.state_dict(), self.target_net_path)
        self.logger.info(f"[step={self.global_step}] Pesos salvos em disco.")

    def __del__(self):
        self.save()


class DQNRouter(Router):
    def __init__(
            self,
            demands,
            state_space: TFRStateSpace,
            learner: Learner,   # 🔹 novo: recebe o learner
            simulation_memory: RailroadEvolutionMemory,
            exploration_method: Callable = lambda x: None,
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
                # demand_index = self.policy_net(state).argmax().item()
                # selected_demand = self.action_space.get_demand(demand_index)
                # Passa o estado pela rede
                q_values = self.policy_net(state).detach().cpu().numpy().flatten()

                # Ordena os índices das ações pelo valor Q (do maior para o menor)
                sorted_indices = q_values.argsort()[::-1]

                # Se quiser, também pode obter os Q-values ordenados:
                sorted_q_values = q_values[sorted_indices]

                # Exemplo: selecionar o melhor, o segundo melhor, etc.
                i = 0
                while i < len(self.action_space.demands):
                    best_action_index = sorted_indices[i]
                    try:
                        # Exemplo: obter a ação correspondente ao melhor Q
                        selected_demand = self.action_space.get_demand(best_action_index)
                        break
                    except:
                        i += 1        

        task = self.demand_to_task(
            selected_demand=selected_demand,
            current_location=current_location,
            train_size=train_size,
            current_time=current_time,
            model_state=model_state,
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
