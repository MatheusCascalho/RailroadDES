from interfaces.train_interface import TrainInterface
from typing import Union
from datetime import datetime, timedelta


class Slot:
    current_train: Union[None, TrainInterface]
    process_start: Union[None, datetime]
    process_end: Union[None, datetime]

    @property
    def is_idle(self):
        return self.current_train is None

    def time_to_be_idle(self, current_time) -> timedelta:
        return timedelta if self.is_idle else self.process_end - current_time

    def put(self, train: TrainInterface, date: datetime, time: timedelta):
        self.current_train = train
        self.process_start = date
        self.process_end = date + time

    def get_out(self):
        self.current_train = None
        self.process_start = None
        self.process_end = None
