from abc import ABC, abstractmethod


class TrainInterface(ABC):
    # ====== Properties ==========
    @property
    @abstractmethod
    def is_empty(self):
        pass

    @property
    @abstractmethod
    def next_location(self):
        pass

    @property
    @abstractmethod
    def volume(self):
        pass

    @volume.setter
    @abstractmethod
    def volume(self, new_volume):
        pass

    @property
    @abstractmethod
    def current_location(self):
        pass

    # @property
    # @abstractmethod
    # def next_process(self):
    #     pass

    # ====== Properties ==========
    # ====== Events ==========
    @abstractmethod
    def load(self, **kwargs):
        pass

    @abstractmethod
    def unload(self, **kwargs):
        pass

    @abstractmethod
    def maneuvering_to_enter(self, **kwargs):
        pass

    @abstractmethod
    def maneuvering_to_leave(self):
        pass

    @abstractmethod
    def arrive(self, **kwargs):
        pass

    @abstractmethod
    def leave(self):
        pass

    # ====== Events ==========
