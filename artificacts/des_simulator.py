import pytest
from abc import abstractmethod
from datetime import datetime


class FakeDESSimulator:
    def __init__(self, time):
        self.time = time

    @abstractmethod
    def add_event(self, **kwargs):
        pass


@pytest.fixture
def fake_des_simulator():
    return FakeDESSimulator(time=datetime(2020,1,1))
