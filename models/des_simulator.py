from interfaces.des_simulator_interface import DESSimulatorInterface
from models import event_calendar as ec
from datetime import datetime, timedelta
from models.exceptions import FinishedTravelException
from models.des_model import DESModel


class DESSimulator(DESSimulatorInterface):
    """
    Discrete Event System simulator
    """
    def __init__(self, initial_date: datetime):
        # setup
        self.calendar = ec.EventCalendar()
        self.current_date = initial_date
        self.initial_date = initial_date

    def add_event(self, time, callback, **data):
        self.calendar.push(time, callback, **data)

    def simulate(self, model: DESModel, time_horizon=timedelta(hours=28)):
        model.starting_events(simulator=self)
        end_date = self.initial_date + time_horizon
        while not self.calendar.is_empty and self.current_date <= end_date:
            # get next event and execute callback
            event = self.calendar.pop()
            self.current_date += event.time_until_happen
            self.calendar.update_events(time_step=event.time_until_happen)
            try:
                event.callback(**event.data)
            except Exception as error:
                model.solver_exceptions(exception=error, event=event)
                event.callback(**event.data)

