from collections import deque

from models.des_model import DESModel
from models.railroad import Railroad
from models.tfr_state_factory import TFRStateFactory, TFRStateSpaceFactory

from logging import critical

class RailroadEvolutionMemory:
    def __init__(self, railroad: Railroad=None, memory_size: int=1000) -> None:
        self._memory = deque(maxlen=memory_size)
        self._railroad = railroad

    @property
    def railroad(self) -> Railroad:
        return self._railroad

    @railroad.setter
    def railroad(self, railroad: Railroad) -> None:
        self._railroad = railroad

    def take_a_snapshot(self, *args, **kwargs):
        if not self.railroad:
            critical("Memory does not know the railroad and therefore does not perform any snapshots")
            return
        state = TFRStateFactory(self.railroad)
        self.memory.append(state)


    @property
    def memory(self):
        return self._memory

    def __repr__(self):
        states = len(self.memory)
        return f"Memory of {states} states of {self.railroad}"

