from models.railroad_mesh import RailSegment
from models.time_table import EventName, TimeEvent
from datetime import datetime, timedelta
from models.arrive_scheduler import ArriveScheduler
from unittest.mock import MagicMock

def test_update_time_table_should_notify_scheduler(time_table, mocker):
    # Arrange
    scheduler = ArriveScheduler([])
    scheduler.update = MagicMock(wraps=scheduler.update)

    time_table.add_observers([scheduler])
    event1 = TimeEvent(event=EventName.DEPARTURE, instant=datetime(2025, 4, 10, 12, 0))

    # Act
    time_table.update(event1)

    # Assert
    scheduler.update.assert_called()

def test_send_should_add_arrive_after_transit_time(
        mocker,
        time_table,
        mock_train,
        basic_node_factory
):
    # Arrange

    origin = mocker.Mock(name='origin')
    destination = mocker.Mock(name='destination')
    segment = RailSegment(
        origin=origin,
        destination=destination,
        time_to_origin=timedelta(hours=5),
        time_to_destination=timedelta(hours=6)
    )
    simulator = MagicMock()
    scheduler = ArriveScheduler(rail_segments=[segment], simulator=simulator)
    scheduler.train = mock_train
    scheduler.update = MagicMock(wraps=scheduler.update)

    time_table.add_observers([scheduler])
    event1 = TimeEvent(event=EventName.DEPARTURE, instant=datetime(2025, 4, 10, 12, 0))

    # Act
    time_table.update(event1)

    # Assert
    simulator.add_event.assert_called_with(
        time=segment.time_to_destination,
        callback=destination.receive,
        train=mock_train
    )


def test_send_should_update_rail_segments(
        mocker,
        time_table,
        basic_node_factory
):
    # Arrange

    origin = mocker.Mock(name='origin')
    destination = mocker.Mock(name='destination')
    segment = RailSegment(
        origin=origin,
        destination=destination,
        time_to_origin=timedelta(hours=5),
        time_to_destination=timedelta(hours=6)
    )
    simulator = MagicMock()
    scheduler = ArriveScheduler(rail_segments=[segment], simulator=simulator)
    scheduler.update = MagicMock(wraps=scheduler.update)

    time_table.add_observers([scheduler])
    event1 = TimeEvent(event=EventName.ARRIVE, instant=datetime(2025, 4, 10, 12, 0))

    # Act
    time_table.update(event1)

    # Assert
    simulator.add_event.assert_not_called()
    assert scheduler.rail_segments == []



