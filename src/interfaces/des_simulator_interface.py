from abc import abstractmethod


class DESSimulatorInterface:
    """
    Discrete Event System simulator
    """
    @abstractmethod
    def add_event(self, time, callback, **data):
        pass

    @abstractmethod
    def simulate(self, model, time_horizon=28 * 3600):
        pass

    @abstractmethod
    def solve_exceptions(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def current_date(self):
        pass
