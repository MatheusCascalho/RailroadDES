import abc
from interfaces.des_simulator_interface import DESSimulatorInterface
from models.event_calendar import Event


class DESModel(abc.ABC):
    def __init__(
            self,
            controllable_events: list[Event],
            uncontrollable_events: list[Event],
    ):
        self.controllable_events = []
        self.uncontrollable_events = []

    @abc.abstractmethod
    def starting_events(self, simulator: DESSimulatorInterface):
        pass

    @abc.abstractmethod
    def solver_exceptions(self, exception: Exception, event: Event):
        pass


