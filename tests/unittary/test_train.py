from models.train import Train
from models.exceptions import TrainExceptions
from datetime import datetime, timedelta
import pytest


def test_train_is_empty():
    train = Train(
        id=1,
        origin=0,
        destination=1,
        model=0,
        path=[0, 1, 0]
    )

    assert train.is_empty


def test_train_should_raise_and_exception_when_load_in_movement(empty_train, fake_des_simulator, fake_node):
    with pytest.raises(TrainExceptions) as t_error:
        empty_train.load(
            simulator=fake_des_simulator,
            volume=4,
            start=datetime(2020, 1, 1),
            process_time=timedelta(hours=5),
            node=fake_node
        )


def test_train_is_empty_when_volume_is_less_than_epsilon(empty_train, fake_des_simulator, fake_node):
    empty_train.arrive(
        simulator=fake_des_simulator,
        node=fake_node
    )

    empty_train.load(
        simulator=fake_des_simulator,
        volume=4,
        start=datetime(2020,1,1),
        process_time=timedelta(hours=5),
        node=fake_node
    )
    empty_train.unload(
        simulator=fake_des_simulator,
        volume=3.99,
        start=datetime(2020,1,1),
        process_time=timedelta(hours=5),
        node=fake_node
    )
    assert empty_train.is_empty


# def test_train_should_update_current_location_and_path_when_arrives():
#     train = Train(
#         id=1,
#         origin=0,
#         destination=1,
#         model=0,
#         current_location=2,
#         eta=datetime(2022, 1, 2),
#         etd=datetime(2022, 1, 1),
#         volume=0.09,
#         path=[1, 0, 1, 2]
#     )
#
#     train.arrive()
#
#     assert train.current_location == 1
#     assert train.path == [0, 1, 2]
#
#
# def test_train_should_return_next_location():
#     train = Train(
#         id=1,
#         origin=0,
#         destination=1,
#         model=0,
#         current_location=2,
#         eta=datetime(2022, 1, 2),
#         etd=datetime(2022, 1, 1),
#         volume=0.09,
#         path=[1, 0, 1, 2]
#     )
#
#     assert train.next_location == 1
#
#
# def test_train_should_raise_an_exception_when_path_is_empty_and_next_location_is_called():
#     train = Train(
#         id=1,
#         origin=0,
#         destination=1,
#         model=0,
#         current_location=2,
#         eta=datetime(2022, 1, 2),
#         etd=datetime(2022, 1, 1),
#         volume=0.09,
#         path=[]
#     )
#     with pytest.raises(TrainExceptions) as t_error:
#         train.next_location
#
#     assert t_error.value.args[0] == 'Path is finished!'
#
#
# def test_train_should_update_volume_when_load_event_is_called(empty_train):
#     empty_train.load(volume=5e3)
#     assert empty_train.volume == 5e3
#
#
# def test_train_should_update_volume_when_load_and_unload_events_are_called(empty_train):
#     empty_train.load(volume=5e3, start=, time_to_load=)
#     empty_train.unload(volume=2e3)
#     assert empty_train.volume == 3e3
