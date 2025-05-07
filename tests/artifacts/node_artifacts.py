import pytest
from models.model_queue import Queue
from models.constants import Process
from models.node import Node


@pytest.fixture
def mock_train(mocker):
    train = mocker.Mock()
    train.ID = "T1"
    train.current_process_name = Process.LOAD
    return train


@pytest.fixture
def mock_simulator(mocker):
    simulator = mocker.Mock()
    simulator.current_date = "2025-01-01"
    return simulator


@pytest.fixture
def mock_slot(mocker):
    slot = mocker.Mock()
    slot.type = Process.LOAD
    slot.is_idle = True
    slot.get_process_time.return_value = 5
    return slot


@pytest.fixture
def basic_node(mocker):
    clock = mocker.Mock()
    clock.current_time = "12:00"
    return Node(
        name="N1",
        clock=clock,
        process_constraints=[],
        maneuvering_constraint_factory=mocker.Mock(),
        queue_to_enter=mocker.Mock(),
        queue_to_leave=mocker.Mock(),
        process_units=[]
    )

@pytest.fixture
def basic_node_factory(mocker):
    def make(name, clock=None):
        clk = mocker.Mock() if not clock else clock
        if not clock:
            clk.current_time = "12:00"
        return Node(
            name=name,
            clock=clk,
            process_constraints=[],
            maneuvering_constraint_factory=mocker.Mock(),
            queue_to_enter=mocker.Mock(),
            queue_to_leave=mocker.Mock(),
            process_units=[]
        )
    return make

@pytest.fixture
def real_node_factory(mocker):
    def make(name, clock=None):
        clk = mocker.Mock() if not clock else clock
        if not clock:
            clk.current_time = "12:00"
        return Node(
            name=name,
            clock=clk,
            process_constraints=[],
            maneuvering_constraint_factory=mocker.Mock(),
            queue_to_enter=Queue(capacity=20, name='input'),
            queue_to_leave=Queue(capacity=20, name='output'),
            process_units=[]
        )
    return make
