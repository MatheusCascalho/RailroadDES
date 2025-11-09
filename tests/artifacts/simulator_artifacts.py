from datetime import datetime, timedelta
from pytest import fixture
from src.simulation.clock import Clock
from src.des_simulator import DESSimulator


class FakeSimulator(DESSimulator):
    def __init__(self, clock: Clock):
        super().__init__(clock=clock)


@fixture
def simple_simulator():
    clk = Clock(
        start=datetime(2025, 4, 1),
        discretization=timedelta(hours=1)
    )
    return FakeSimulator(clk)