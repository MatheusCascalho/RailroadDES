import pytest
from unittest.mock import MagicMock, call, patch
from models.node import Node
from models.clock import Clock
from datetime import datetime, timedelta
from typing import Callable

@pytest.fixture
def current_time():
    t = datetime(2025, 5, 1, 10, 0)
    return t

@pytest.fixture
def mock_clock(current_time):
    clock = MagicMock(spec=Clock)
    clock.current_time = current_time  # Mocking the current time
    return clock

@pytest.fixture
def fake_queue():
    def gen():
        t1 = MagicMock()
        t1.current_process_name = "load"
        yield t1
        t2 = MagicMock()
        t2.current_process_name = "load"
        yield t2
    return gen

@pytest.fixture
def free_constraint():
    def make():
        constraint = MagicMock()
        constraint.is_blocked.return_value = True
        return constraint
    return make

@pytest.fixture
def blocked_constraint():
    def make():
        constraint = MagicMock()
        constraint.is_blocked.return_value = False
        return constraint
    return make

@pytest.fixture
def constraint_list(free_constraint, blocked_constraint):
    def make(free: int = 1, blocked: int = 0):
        f = [free_constraint() for _ in range(free)]
        b = [blocked_constraint() for _ in range(blocked)]
        c = f + b
        return c
    return make

@pytest.fixture
def block_constraints(constraint_list):
    def wrapper():
        result = constraint_list(free=0,blocked=1)
        return result
    return wrapper


@pytest.fixture
def node_with_mocks(mock_clock, fake_queue, constraint_list):
    def make(
            queue_generator=fake_queue,
            process_constraints_generator: Callable =constraint_list,
    ):
        queue = MagicMock()
        queue.running_queue.return_value = queue_generator()
        constraints = process_constraints_generator()

        slot = MagicMock()
        slot.type = 'load'
        slot.is_idle = True

        return Node(
            name="Node1",
            clock=mock_clock,
            process_constraints=constraints,
            maneuvering_constraint_factory=MagicMock(),
            process_units=[slot],
            queue_to_leave=queue,
            queue_to_enter=queue
        )
    return make

@pytest.fixture
def mock_train():
    train = MagicMock()
    train.ID = "Train1"
    train.next_location = "Node2"
    train.current_process_name = 'load'
    return train



def test_receive_train(node_with_mocks, mock_train, current_time):
    # Testando o recebimento de um trem no nó
    node = node_with_mocks()
    node.receive(mock_train)

    ## Assert that the input queue's push method has been executed
    node.queue_to_enter.push.assert_called()


@pytest.mark.timeout(0.5)
def test_blocked_constraint_should_prevent_pop_queue(
        node_with_mocks,
        mock_train,
        block_constraints
):
    # Testando o caso onde o trem não pode continuar sua viagem
    node = node_with_mocks(
        process_constraints_generator=block_constraints
    )
    sim = MagicMock()
    node.process(sim)
    node.queue_to_enter.pop.assert_not_called()


