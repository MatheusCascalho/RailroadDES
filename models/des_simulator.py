from models import event_calendar as ec
from datetime import datetime


class DESSimulator:
    """
    Discrete Event System simulator
    """
    def __init__(self):
        # setup
        self.calendar = ec.EventCalendar()
        self.time = datetime

    def add_event(self, time, callback, **data):
        self.calendar.push(time, callback, **data)

    def simulate(self, model, time_horizon=28 * 3600):
        model.starting_events()
        while not self.calendar.is_empty and self.time <= time_horizon:
            # get next event and execute callback
            self.time, callback, data = self.calendar.pop()
            callback(self, data)
