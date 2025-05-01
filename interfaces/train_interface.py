from abc import ABC, abstractmethod


class TrainInterface(ABC):
    # ====== Properties ==========

    @property
    @abstractmethod
    def current_task(self):
        pass

    @current_task.setter
    @abstractmethod
    def current_task(self, value):
        pass

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
    def capacity(self):
        pass

    @property
    @abstractmethod
    def product(self):
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

    @property
    @abstractmethod
    def process_end(self):
        pass


    @property
    @abstractmethod
    def current_process_name(self):
        pass

    @property
    @abstractmethod
    def state(self):
        pass

    @property
    @abstractmethod
    def ID(self):
        pass

    @property
    @abstractmethod
    def ready_to_leave(self):
        pass


    @abstractmethod
    def add_to_slot(self):
        pass

    @abstractmethod
    def removed_from_slot(self):
        pass

    # ====== Properties ==========
    # ====== Events ==========
    @abstractmethod
    def start_load(self, **kwargs):
        pass

    @abstractmethod
    def start_unload(self, **kwargs):
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
    def leave(self, **kwargs):
        pass

    # ====== Events ==========
