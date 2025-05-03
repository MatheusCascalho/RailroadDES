from clock import Clock
from abc import abstractmethod

def entity_id_gen():
    i = 0
    while True:
        ID = f"entity_{i}"
        yield ID
        i += 1

entity_id = entity_id_gen()

class Entity:
    def __init__(self, name, clock: Clock):
        self._id = next(entity_id)
        self.name = name
        self.clock = clock

    @property
    def identifier(self):
        return self._id

    @abstractmethod
    @property
    def state(self):
        pass

    def __repr__(self):
        return self.name

    __str__ = __repr__