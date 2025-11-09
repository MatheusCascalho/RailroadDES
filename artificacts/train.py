from src.train import Train
import pytest


@pytest.fixture
def empty_train():
    train = Train(
        id=1,
        origin=0,
        destination=1,
        model=0,
        path=[0,1,0]
    )
    return train
