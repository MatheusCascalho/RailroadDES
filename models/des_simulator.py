from interfaces.des_simulator_interface import DESSimulatorInterface
from models import event_calendar as ec
from datetime import datetime, timedelta
from models.exceptions import FinishedTravelException
from models.des_model import DESModel
from models.clock import Clock
from models.entity import Entity


class DESSimulator(Entity, DESSimulatorInterface):
    """
    Discrete Event System simulator
    """
    def __init__(self, clock: Clock):
        # setup
        self.calendar = ec.EventCalendar()
        self.clock = clock
        self.initial_date = clock.current_time
        self.model = None ## Melhorar isso!
        super().__init__(name='sim', clock=clock)

    @property
    def current_date(self):
        return self.clock.current_time

    def add_event(self, time, callback, **data):
        self.calendar.push(time, callback, **data)

    def simulate(self, model: DESModel, time_horizon=timedelta(hours=28)):
        self.model = model
        self.model.starting_events(simulator=self)
        end_date = self.initial_date + time_horizon
        while not self.calendar.is_empty and self.current_date <= end_date:
            # get next event and execute callback
            event = self.calendar.pop()
            self.clock.jump(event.time_until_happen)
            self.calendar.update_events(time_step=event.time_until_happen)
            try:
                event.callback(**event.data)
            except FinishedTravelException as error:
                self.model.solver_exceptions(exception=error, event=event, simulator=self)
                event.callback(**event.data)

    def solve_exceptions(self, *args, **kwargs):
        error = kwargs.get('error')
        event = kwargs.get('event')
        self.model.solver_exceptions(exception=error, event=event, simulator=self)

