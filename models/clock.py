from datetime import datetime, timedelta

class Clock:
    def __init__(self, start: datetime, discretization: timedelta):
        self.__current_time = start
        self.discretization = discretization

    @property
    def current_time(self):
        return self.__current_time

    def update(self):
        self.__current_time += self.discretization

    def jump(self, step: timedelta):
        self.__current_time += step