import pytest
from abc import abstractmethod
from datetime import timedelta
from interfaces.node_interce import NodeInterface


class FakeNode(NodeInterface):
    @property
    def time_to_call(self):
        return timedelta()

    def call_to_enter(self, **kwargs):
        pass

    def process(self, **kwargs):
        pass

    def maneuver_to_dispatch(self, **kwargs):
        pass

    def __init__(self):
        pass


@pytest.fixture
def fake_node():
    return FakeNode()
