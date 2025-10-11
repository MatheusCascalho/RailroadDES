from collections import deque
from dataclasses import dataclass, field
from queue import Full
from typing import Callable
from models.des_model import DESModel
from models.observers import AbstractSubject, to_notify, AbstractObserver
from models.railroad import Railroad
from models.states import ActivityState
from models.tfr_state_factory import TFRStateFactory, TFRState
from models.demand import Flow
from multiprocessing import Queue
import dill
import os
from logging import critical
from models.pickle_debugger import find_pickle_issues as find_unpicklables
import logging
import numpy as np

VERBOSE = False


def memory_id_gen():
    i = 0
    while True:
        mem_id = f"Memory{i} - PID {os.getpid()}"
        yield mem_id
        i += 1

memory_id = memory_id_gen()

@dataclass(frozen=True)
class Experience:
    state: TFRState
    action: str
    reward: float
    next_state: TFRState
    is_done: bool
    memory_id: str = field(default_factory=lambda: next(memory_id))

    def __iter__(self):
        values = [self.state, self.action, self.reward, self.next_state, self.is_done]
        return iter(values)

    def is_static(self):
        s1 = str(self.state)
        s2 = str(self.next_state)
        return s1 == s2

    def __eq__(self, other):
        return self.memory_id == other.memory_id


class RailroadEvolutionMemory(AbstractSubject):
    def __init__(self, railroad: Railroad=None, memory_size: int=1000, state_factory: Callable = TFRStateFactory) -> None:
        self._memory = deque(maxlen=memory_size)
        self._railroad = railroad
        self.previous_state = None
        self.state_factory = state_factory
        super().__init__()
        self.initial_time = None
        # Configuração de logger
        self.mem_id = next(memory_id)
        self.cumulated_reward = 0 
        self.logger = logging.getLogger(f'memory_{memory_id}')
        self.logger.setLevel(logging.INFO)
        # evita múltiplos handlers duplicados
        if not self.logger.handlers:
            log_dir = "logs/memory"
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(log_dir, f"memory.log"))
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            sh = logging.StreamHandler()
            self.logger.addHandler(sh)

    @property
    def railroad(self) -> Railroad:
        return self._railroad

    @railroad.setter
    def railroad(self, railroad: Railroad) -> None:
        self._railroad = railroad
        self.initial_time = self.railroad.mesh.load_points[0].clock.current_time

    def save_previous_state(self, *args, **kwargs):
        state = self.take_a_snapshot(is_initial=kwargs.get('is_initial', False))
        self.previous_state = state


    def take_a_snapshot(self, *args, **kwargs) -> TFRState:
        is_initial = kwargs.get('is_initial', False)
        if not self.railroad:
            critical("Memory does not know the railroad and therefore does not perform any snapshots")
            return
        state = self.state_factory(railroad=self.railroad, is_initial=is_initial)
        return state

    def save_consequence(self, *args, **kwargs):
        event_name = kwargs.get("event_name", "AUTOMATIC")
        next_state = self.take_a_snapshot(*args, **kwargs)
        self.save(
            s1=self.previous_state,
            s2=next_state,
            a=event_name,
            r=next_state.reward(),
        )


    @property
    def memory(self):
        return self._memory

    @to_notify()
    def save(self, s1: TFRState, a, r: float, s2: TFRState):
        element = Experience(state=s1, action=a, reward=r, next_state=s2, is_done=s2.is_final)
        if not element.is_static():
            self._memory.append(element)

            opvol = sum([d.operated for d in self.railroad.demands])
            demand = sum([d.volume for d in self.railroad.demands])
            queues = np.median(
                [
                    t.penalty().total_seconds()/(60*60) 
                    for t, train in self.railroad.router.running_tasks.items() 
                    if train.current_activity.name in [ActivityState.QUEUE_TO_ENTER, ActivityState.QUEUE_TO_LEAVE]
                ]
            )
            balance = element.next_state.railroad_balance()
            self.cumulated_reward += element.reward
            simulation_ellapsed = (self.railroad.mesh.load_points[0].clock.current_time - self.initial_time).total_seconds()/(60*60)
            if VERBOSE:
                self.logger.info(
                    f"mem_id={self.mem_id} | "
                    f"mem_size={len(self._memory)} | "
                    f"queues[h]={queues:.4f} | "
                    f"current_reward={element.reward:.5f} | "
                    f"cumulated_reward={self.cumulated_reward:.5f} | "
                    f"operated_volume={opvol:.2f} | "
                    f"demand={demand:.2f} | "
                    f"balance={balance} | "
                    f"simulation_time[h]={simulation_ellapsed:.2f}"
                )


    def __repr__(self):
        states = len(self.memory)
        return f"Memory of {states} states of {self.railroad}"

    __str__ = __repr__

    @property
    def last_item(self):
        if self.memory:
            return self.memory[-1]
        return None
    
    @property
    def next_state(self):
        if self.last_item:
            return self.last_item.next_state

    def __iter__(self):
        return self.memory.__iter__()

class ExperienceProducer(AbstractObserver):
    def __init__(self, queue, memory_size: int=100_000):
        self._memory = deque(maxlen=memory_size)
        self.queue = queue
        self.existing_keys = set()
        super().__init__()

    def update(self):
        experience = self.subjects[0].last_item
        if experience and self._experience_key(experience) not in self.existing_keys:
            self.queue.put(experience, timeout=1)
            self.existing_keys.add(self._experience_key(experience=experience))

        self._memory.append(experience)

    def _experience_key(self, experience):
        return experience.memory_id

    @property
    def memory(self):
        return self._memory
