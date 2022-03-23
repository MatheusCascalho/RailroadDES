from abc import ABC, abstractmethod


class TrainInterface(ABC):
    # ====== Properties ==========
    # @abstractmethod
    # @property
    # def is_empty(self):
    #     pass
    #
    # @abstractmethod
    # @property
    # def next_location(self):
    #     pass
    #
    # @abstractmethod
    # @property
    # def volume(self):
    #     pass
    #
    # @abstractmethod
    # @volume.setter
    # def volume(self, new_volume):
    #     pass
    #
    # @abstractmethod
    # @property
    # def current_location(self):
    #     pass
    #
    # @abstractmethod
    # @property
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
