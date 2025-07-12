from collections import deque
from dataclasses import dataclass
from models.des_model import DESModel
from models.observers import AbstractSubject, to_notify
from models.railroad import Railroad
from models.tfr_state_factory import TFRStateFactory, TFRState
from models.demand import Flow

from logging import critical


@dataclass(frozen=True)
class MemoryElement:
    state: TFRState
    action: str
    reward: float
    next_state: TFRState
    is_done: bool

    def __iter__(self):
        values = [self.state, self.action, self.reward, self.next_state, self.is_done]
        return iter(values)


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
        state = self.take_a_snapshot()
        self.previous_state = state


    def take_a_snapshot(self, *args, **kwargs):
        if not self.railroad:
            critical("Memory does not know the railroad and therefore does not perform any snapshots")
            return
        state = TFRStateFactory(self.railroad)
        return state

    def save_consequence(self, *args, **kwargs):
        event_name = kwargs.get("event_name", "AUTOMATIC")
        state = self.take_a_snapshot(*args, **kwargs)
        self.save(
            s1=self.previous_state,
            s2=state,
            a=event_name,
            r=state.reward(),
        )


    @property
    def memory(self):
        return self._memory

    @to_notify()
    def save(self, s1: TFRState, a, r: float, s2: TFRState):
        element = MemoryElement(state=s1, action=a, reward=r, next_state=s2, is_done=s2.is_final)
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

    def __iter__(self):
        return self.memory.__iter__()

