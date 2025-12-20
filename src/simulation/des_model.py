import abc
from calendar import Calendar
from interfaces.des_simulator_interface import DESSimulatorInterface
from src.events.event import Event
from datetime import timedelta
from src.events.event_chain import EventChain
from src.events.event_queue import EventQueue
from src.simulation.clock import Clock
from src.simulation.event_calendar import EventCalendar


class DESModel(abc.ABC):
    def __init__(
            self,
            calendar: EventCalendar
    ):
        self._calendar = calendar
        self._event_queue = EventQueue()
        self._event_queue.add_observers([self._calendar])

    @abc.abstractmethod
    def starting_events(self, simulator: DESSimulatorInterface, time_horizon: timedelta):
        pass

    @abc.abstractmethod
    def solver_exceptions(self, exception: Exception, event: Event, simulator: DESSimulatorInterface):
        pass

    @abc.abstractmethod
    def model_clocks(self) -> list[Clock]:
        pass

    @abc.abstractmethod
    def stop_criteria(self):
        """
        Criterio de parada em simulações
        """
        pass


