from models.entities import Train, Node
from models.exceptions import TrainExceptions
from datetime import datetime
import pytest


def test_train_is_empty():
    train = Train(
        id=1,
        origin=0,
        destination=1,
        model=0,
        current_location=0,
        eta=datetime(2022, 1, 2),
        etd=datetime(2022, 1, 1),
        volume=0.0,
        path=[]
    )

    assert train.is_empty


def test_train_is_not_empty():
    train = Train(
        id=1,
        origin=0,
        destination=1,
        model=0,
        current_location=0,
        eta=datetime(2022, 1, 2),
        etd=datetime(2022, 1, 1),
        volume=5e3,
        path=[]
    )

    assert not train.is_empty


def test_train_is_empty_when_volume_is_less_than_epsilon():
    train = Train(
        id=1,
        origin=0,
        destination=1,
        model=0,
        current_location=0,
        eta=datetime(2022, 1, 2),
        etd=datetime(2022, 1, 1),
        volume=0.09,
        path=[]
    )

    assert train.is_empty


def test_train_should_update_current_location_and_path_when_arrives():
    train = Train(
        id=1,
        origin=0,
        destination=1,
        model=0,
        current_location=2,
        eta=datetime(2022, 1, 2),
        etd=datetime(2022, 1, 1),
        volume=0.09,
        path=[1, 0, 1, 2]
    )

    train.arrive()

    assert train.current_location == 1
    assert train.path == [0, 1, 2]


def test_train_should_return_next_location():
    train = Train(
        id=1,
        origin=0,
        destination=1,
        model=0,
        current_location=2,
        eta=datetime(2022, 1, 2),
        etd=datetime(2022, 1, 1),
        volume=0.09,
        path=[1, 0, 1, 2]
    )

    assert train.next_location == 1


def test_train_should_raise_an_exception_when_path_is_empty_and_next_location_is_called():
    train = Train(
        id=1,
        origin=0,
        destination=1,
        model=0,
        current_location=2,
        eta=datetime(2022, 1, 2),
        etd=datetime(2022, 1, 1),
        volume=0.09,
        path=[]
    )
    with pytest.raises(TrainExceptions) as t_error:
        train.next_location

    assert t_error.value.args[0] == 'Path is finished!'
