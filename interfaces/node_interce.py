from abc import ABC, abstractmethod


class NodeInterface(ABC):
    # ====== Properties ==========
    @property
    @abstractmethod
    def time_to_call(self):
        pass

    # @property
    # def identifier(self):
    #     pass
    #
    # @abstractmethod
    # @identifier.setter
    # def identifier(self, **kwargs):
    #     pass
    #
    # @abstractmethod
    # @property
    # def process_time(self):
    #     pass
    #
    # @abstractmethod
    # @property
    # def processing_slots(self):
    #     pass
    #
    # @abstractmethod
    # def next_idle_slot(self, current_time):
    #     pass

    # ====== Properties ==========
    # ====== Events ==========
    @abstractmethod
    def call_to_enter(self, **kwargs):
        pass

    @abstractmethod
    def process(self, **kwargs):
        pass

    @abstractmethod
    def maneuver_to_dispatch(self, **kwargs):
        pass
