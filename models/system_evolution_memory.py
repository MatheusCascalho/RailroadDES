from collections import deque
from dataclasses import dataclass, field
from queue import Full

from models.des_model import DESModel
from models.observers import AbstractSubject, to_notify, AbstractObserver
from models.railroad import Railroad
from models.tfr_state_factory import TFRStateFactory, TFRState
from models.demand import Flow
from multiprocessing import Queue
import dill
import os
from logging import critical
from models.pickle_debugger import find_pickle_issues as find_unpicklables

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
    def __init__(self, railroad: Railroad=None, memory_size: int=1000) -> None:
        self._memory = deque(maxlen=memory_size)
        self._railroad = railroad
        self.previous_state = None
        super().__init__()

    @property
    def railroad(self) -> Railroad:
        return self._railroad

    @railroad.setter
    def railroad(self, railroad: Railroad) -> None:
        self._railroad = railroad

    def save_previous_state(self, *args, **kwargs):
        state = self.take_a_snapshot(is_initial=kwargs.get('is_initial', False))
        self.previous_state = state


    def take_a_snapshot(self, *args, **kwargs) -> TFRState:
        is_initial = kwargs.get('is_initial', False)
        if not self.railroad:
            critical("Memory does not know the railroad and therefore does not perform any snapshots")
            return
        state = TFRStateFactory(self.railroad, is_initial=is_initial)
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
