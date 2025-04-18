import pytest
from datetime import datetime, timedelta
from models.constants import Process, EventName
from models.time_table import (
    TimeTable,
    TimeEvent,
    TimeRegister
)
from models.exceptions import EventSequenceError, TimeSequenceErro, RepeatedProcessError, AlreadyRegisteredError


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


def test_time_table_initialization(time_table):
    """Test that TimeTable initializes with an empty list of registers."""
    assert len(time_table.registers) == 0


def test_update_register_with_event(time_table):
    """Test that an event is correctly added to a new register."""
    event = TimeEvent(event=EventName.ARRIVE, instant=datetime(2025, 4, 10, 12, 0))
    time_table.update(event)

    assert len(time_table.registers) == 1
    assert time_table.registers[0].arrive == event.instant


def test_update_existing_register(time_table):
    """Test that an event is correctly added to the last register."""
    event1 = TimeEvent(event=EventName.ARRIVE, instant=datetime(2025, 4, 10, 12, 0))
    event2 = TimeEvent(event=EventName.START_PROCESS, instant=datetime(2025, 4, 10, 12, 15))

    time_table.update(event1)
    time_table.update(event2)

    assert time_table.registers[0].start_process == event2.instant


def test_calculate_queue_time(time_table, time_register):
    """Test that queue time is calculated correctly."""
    # Setting up the time register
    time_table.registers.append(time_register)

    # Calculating expected queue time
    expected_queue_time = timedelta(minutes=15) + timedelta(
        minutes=15)  # 15 minutes from arrive to start_process + 15 minutes from finish_process to departure
    assert time_table.queue_time == expected_queue_time


def test_calculate_transit_time(time_table):
    """Test that transit time is calculated correctly with only 2 registers (arrival and departure)."""
    # Departure event (previous register) and Arrival event (current register)
    event1 = TimeEvent(event=EventName.DEPARTURE, instant=datetime(2025, 4, 10, 13, 0))  # Departure 1 hour later
    event2 = TimeEvent(event=EventName.ARRIVE, instant=datetime(2025, 4, 11, 13, 0))  # Arrival 1 day later

    # Update the time table with the departure and arrival events
    time_table.update(event1, process=Process.UNLOAD)
    time_table.update(event2, process=Process.LOAD)

    # Transit time should be 1 day (departure of previous register to arrival of current register)
    expected_transit_time = timedelta(days=1)

    # Assert that the calculated transit time matches the expected transit time
    assert time_table.in_transit_time == expected_transit_time

def test_calculate_util_time(time_table, time_register):
    """Test that utilization time is calculated correctly."""
    # Adding a time register
    time_table.registers.append(time_register)

    # Calculating expected utilization time
    expected_util_time = timedelta(minutes=30)  # 1 hour transit + 30 minutes process time
    assert time_table.util_time == expected_util_time


def test_not_initial_register(time_table):
    """Test that the initial register is correctly identified."""
    event1 = TimeEvent(event=EventName.ARRIVE, instant=datetime(2025, 4, 10, 12, 0))
    time_table.update(event1)

    event2 = TimeEvent(event=EventName.START_PROCESS, instant=datetime(2025, 4, 10, 12, 15))
    time_table.update(event2)

    event3 = TimeEvent(event=EventName.FINISH_PROCESS, instant=datetime(2025, 4, 10, 12, 45))
    time_table.update(event3)

    event4 = TimeEvent(event=EventName.DEPARTURE, instant=datetime(2025, 4, 10, 13, 0))
    time_table.update(event4)

    # Checking if all events are registered on the same register
    assert len(time_table.registers) == 1
    # Checking if the initial register is correctly identified
    assert time_table.registers[0].is_initial_register is False


def test_initial_register(time_table):
    """Test that the initial register is correctly identified."""
    event2 = TimeEvent(event=EventName.START_PROCESS, instant=datetime(2025, 4, 10, 12, 15))
    time_table.update(event2)

    event3 = TimeEvent(event=EventName.FINISH_PROCESS, instant=datetime(2025, 4, 10, 12, 45))
    time_table.update(event3)

    event4 = TimeEvent(event=EventName.DEPARTURE, instant=datetime(2025, 4, 10, 13, 0))
    time_table.update(event4)

    # Checking if all events are registered on the same register
    assert len(time_table.registers) == 1
    # Checking if the initial register is correctly identified
    assert time_table.registers[0].is_initial_register

def test_update_event_when_already_exists(time_table):
    """Test that an exception is raised when trying to update an already existing event."""
    event1 = TimeEvent(event=EventName.DEPARTURE, instant=datetime(2025, 4, 10, 12, 0))
    time_table.update(event1)

    event2 = TimeEvent(event=EventName.DEPARTURE,
                       instant=datetime(2025, 4, 10, 12, 10))  # Trying to update ARRIVE event again
    with pytest.raises(AlreadyRegisteredError, match="Event already registered"):
        time_table.update(event2)

def test_update_should_raise_when_arrive_is_before_departure(time_table):
    """Test that an exception is raised when trying to update an already existing event."""
    event1 = TimeEvent(event=EventName.DEPARTURE, instant=datetime(2025, 4, 10, 12, 0))
    time_table.update(event1, process=Process.LOAD)

    event2 = TimeEvent(event=EventName.ARRIVE,
                       instant=datetime(2025, 4, 9, 12, 10))  # Trying to update ARRIVE event again
    with pytest.raises(TimeSequenceErro):
        time_table.update(event2, process=Process.UNLOAD)

def test_update_should_raise_when_start_is_before_arrive(time_table):
    """Test that an exception is raised when trying to update an already existing event."""
    event1 = TimeEvent(event=EventName.ARRIVE, instant=datetime(2025, 4, 10, 12, 0))
    time_table.update(event1)

    event2 = TimeEvent(event=EventName.START_PROCESS,
                       instant=datetime(2025, 4, 9, 12, 10))  # Trying to update ARRIVE event again
    with pytest.raises(TimeSequenceErro):
        time_table.update(event2)

def test_update_should_raise_when_event_is_from_new_process_and_current_process_is_not_finished(time_table):
    """Test that an exception is raised when trying to update an already existing event."""
    event1 = TimeEvent(event=EventName.ARRIVE, instant=datetime(2025, 4, 10, 12, 0))
    time_table.update(event1, process=Process.UNLOAD)

    event2 = TimeEvent(event=EventName.ARRIVE,
                       instant=datetime(2025, 4, 9, 12, 10))  # Trying to update ARRIVE event again
    with pytest.raises(TimeSequenceErro):
        time_table.update(event2, process=Process.LOAD)

def test_update_should_raise_when_process_is_not_starting_with_arrive(time_table):
    """Test that an exception is raised when trying to update an already existing event."""
    event1 = TimeEvent(event=EventName.DEPARTURE, instant=datetime(2025, 4, 10, 12, 0))
    time_table.update(event1, process=Process.UNLOAD)

    event2 = TimeEvent(event=EventName.START_PROCESS,
                       instant=datetime(2025, 4, 9, 12, 10))  # Trying to update ARRIVE event again
    with pytest.raises(TimeSequenceErro):
        time_table.update(event2, process=Process.LOAD)

def test_update_should_raise_when_process_is_repeated(time_table):
    """Test that an exception is raised when trying to update an already existing event."""
    event1 = TimeEvent(event=EventName.DEPARTURE, instant=datetime(2025, 4, 10, 12, 0))
    time_table.update(event1, process=Process.UNLOAD)

    event2 = TimeEvent(event=EventName.ARRIVE,
                       instant=datetime(2025, 4, 9, 12, 10))  # Trying to update ARRIVE event again
    with pytest.raises(RepeatedProcessError):
        time_table.update(event2, process=Process.UNLOAD)