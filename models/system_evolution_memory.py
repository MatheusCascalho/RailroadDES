from collections import deque

from models.des_model import DESModel
from models.railroad import Railroad
from models.tfr_state_factory import TFRStateFactory, TFRStateSpaceFactory


class RailroadEvolutionMemory:
    def __init__(self, railroad: Railroad, memory_size: int) -> None:
        self._memory = deque(maxlen=memory_size)
        self.railroad = railroad

    def take_a_snapshot(self):
        state = TFRStateFactory(self.railroad)
        self.memory.append(state)

    @property
    def memory(self):
        return self._memory

