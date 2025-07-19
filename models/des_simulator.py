from interfaces.des_simulator_interface import DESSimulatorInterface
from models.event_calendar import EventCalendar
from datetime import datetime, timedelta
from models.exceptions import FinishedTravelException, NotCompletedEvent
from models.des_model import DESModel
from models.clock import Clock
from models.entity import Entity
from logging import debug

class DESSimulator(Entity, DESSimulatorInterface):
    """
    Discrete Event System simulator
    """
    def __init__(self, clock: Clock, calendar: EventCalendar = EventCalendar()):
        # setup
        self.calendar = calendar
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
        self.validate_clock(model=model)
        self.model = model
        self.model.starting_events(simulator=self, time_horizon=time_horizon)
        debug(
            f"Start simulation from {self.initial_date} to {self.initial_date + time_horizon}"
        )
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
            except NotCompletedEvent as error:
                event.reschedule(time_to_happen=timedelta(hours=1))
                self.calendar.push(time=timedelta(hours=1), event=event, callback=None)
        debug(f"Finish simulation")

    def solve_exceptions(self, *args, **kwargs):
        error = kwargs.get('error')
        event = kwargs.get('event')
        self.model.solver_exceptions(exception=error, event=event, simulator=self)

    def validate_clock(self, model: DESModel):
        clocks = model.model_clocks()
        all_clocks_are_the_same = all(c.ID == self.clock.ID for c in clocks)
        if not all_clocks_are_the_same:
            raise Exception(f"Clocks {self.clock} are not the same")
