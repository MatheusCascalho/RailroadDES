from src.domain.entities.time_table import (
    TimeTable,
    TimeEvent,
    TimeRegister
)
from src.domain.constants import Process, EventName
from datetime import datetime
import pytest

@pytest.fixture
def time_table():
    """Fixture to initialize an empty TimeTable."""
    return TimeTable()


@pytest.fixture
def time_register():
    """Fixture to initialize a TimeRegister with example data."""
    arrive = TimeEvent(
        EventName.ARRIVE,
        instant=datetime(2025, 4, 10, 12, 0)
    )
    start = TimeEvent(EventName.START_PROCESS, datetime(2025, 4, 10, 12, 15))
    finish = TimeEvent(EventName.FINISH_PROCESS, datetime(2025, 4, 10, 12, 45))
    departure = TimeEvent(EventName.DEPARTURE, datetime(2025, 4, 10, 13, 0))
    time_register = TimeRegister(
        process=Process.LOAD,
        arrive=arrive,
        start_process=start,
        finish_process=finish,
        departure=departure
    )
    return time_register
