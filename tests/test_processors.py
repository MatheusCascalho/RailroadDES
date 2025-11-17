import pytest
from unittest.mock import MagicMock
from datetime import timedelta, datetime
from src.domain.entities import model_queue as mq
from src.simulation.clock import Clock
from src.domain.constants import Process
from src.domain.entities.node_data_model import ProcessorRate
from src.domain.constraints.stock_constraints import StockToLoadTrainConstraint, StockToUnloadTrainConstraint
from src.domain.exceptions import ProcessException
from src.domain.systems.processors import ProcessorSystem, ProcessorSystemBuilder
from src.domain.states import ProcessorState


@pytest.fixture
def setup_processor_system():
    # Mock para o relógio
    clock = MagicMock(spec=Clock)
    clock.current_time = datetime(2025,1,1)  # Valor de tempo fictício
    # Mock para a fila de saída
    queue_to_leave = MagicMock(spec=mq.Queue)
    # Taxas de processamento fictícias
    rates = {
        'product1': ProcessorRate(rate=10, discretization=timedelta(hours=5), product='product1', type=Process.LOAD),
        'product2': ProcessorRate(rate=20, discretization=timedelta(hours=2), product='product2', type=Process.LOAD)
    }
    # Instância de ProcessorSystem
    processor = ProcessorSystem(
        processor_type=Process.LOAD,
        queue_to_leave=queue_to_leave,
        clock=clock,
        rates=rates
    )
    return processor, clock, queue_to_leave


def test_processor_initialization(setup_processor_system):
    processor, _, _ = setup_processor_system

    # Testa se o ProcessorSystem é inicializado corretamente
    assert processor.ID is not None
    assert processor.type == Process.LOAD
    assert processor.current_train is None
    assert processor.promised is False


def test_put_in_processor_idle(setup_processor_system):
    processor, _, _ = setup_processor_system
    # Mock de trem
    train_mock = MagicMock()
    train_mock.capacity = 100
    processor.state_machine.current_state = ProcessorState.IDLE

    # Mock das restrições para não bloquear o processamento
    processor.active_constraints = MagicMock(return_value=[])

    processor.put_in(train_mock)

    assert processor.current_train == train_mock
    assert processor.state_machine.current_state.name == ProcessorState.BUSY


def test_put_in_processor_busy(setup_processor_system):
    processor, _, _ = setup_processor_system
    # Mock de trem
    train_mock = MagicMock()
    train_mock.capacity = 100
    processor.state_machine.current_state = ProcessorState.BUSY

    with pytest.raises(ProcessException):
        processor.put_in(train_mock)


def test_free_up_processor(setup_processor_system):
    processor, _, _ = setup_processor_system
    train_mock = MagicMock()
    processor.current_train = train_mock
    processor.state_machine.current_state = ProcessorState.BUSY

    freed_train = processor.free_up()

    assert freed_train == train_mock
    assert processor.state_machine.current_state.name == ProcessorState.IDLE


def test_clear_processor(setup_processor_system):
    processor, _, queue_to_leave = setup_processor_system
    train_mock = MagicMock()
    processor.current_train = train_mock
    processor.state_machine.current_state = ProcessorState.BUSY

    cleared_train = processor.clear()

    assert cleared_train == train_mock
    queue_to_leave.push.assert_called_once_with(train_mock, arrive=processor.clock.current_time)


def test_is_ready_to_clear(setup_processor_system):
    processor, _, _ = setup_processor_system
    train_mock = MagicMock()
    train_mock.process_end = datetime(2025,1,1)
    processor.current_train = train_mock

    assert processor.is_ready_to_clear()


def test_get_process_time(setup_processor_system):
    processor, _, _ = setup_processor_system
    train_mock = MagicMock()
    train_mock.capacity = 100
    train_mock.product = 'product1'
    processor.current_train = train_mock
    processor.state_machine.current_state = ProcessorState.BUSY

    process_time = processor.get_process_time()

    expected_time = timedelta(hours=50)  # (100/10) * 5
    assert process_time == expected_time


def test_promise_processor(setup_processor_system):
    processor, _, _ = setup_processor_system
    train_mock = MagicMock()
    train_mock.capacity = 100
    train_mock.product = 'product1'
    processor.current_train = train_mock
    processor.state_machine.current_state = ProcessorState.BUSY

    event_promise = processor.promise()

    assert event_promise.volume == 100
    assert event_promise.event.name == 'DISPATCH_VOLUME'
    assert processor.promised is True


def test_add_constraint(setup_processor_system):
    processor, _, _ = setup_processor_system
    constraint_mock = MagicMock()
    processor.add_constraint(constraint_mock)

    assert constraint_mock in processor.constraints

