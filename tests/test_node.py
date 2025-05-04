from models.constants import Process
from tests.artifacts.node_artifacts import mock_train, mock_simulator, mock_slot, basic_node


# ========== TESTES ==========
def test_receive_pushes_train(basic_node, mock_train, mocker):
    # Arrange
    basic_node.queue_to_enter = mocker.Mock()

    # Act
    basic_node.receive(mock_train)

    # Assert
    basic_node.queue_to_enter.push.assert_called_once_with(mock_train, arrive="12:00")


def test_process_starts_train_process(basic_node, mock_train, mock_slot, mock_simulator, mocker):
    # Arrange
    mock_queue = mocker.Mock()
    mock_queue.running_queue.return_value = [mock_train]
    mock_queue.pop.return_value = mock_train
    mock_queue.first = mock_train
    basic_node.queue_to_enter = mock_queue
    basic_node.process_units = [mock_slot]

    # Act
    basic_node.process(simulator=mock_simulator)

    # Assert
    mock_queue.pop.assert_called_once_with(current_time=mock_simulator.current_date)
    mock_slot.put_in.assert_called_once_with(train=mock_train)
    mock_simulator.add_event.assert_called_once()
    mock_queue.skip_process.assert_not_called()  # Verifica que não pulou


def test_process_skips_when_slot_unavailable(basic_node, mock_train, mock_simulator, mocker):
    # Arrange
    mock_queue = mocker.Mock()
    mock_queue.running_queue.return_value = [mock_train]
    basic_node.queue_to_enter = mock_queue
    basic_node.process_units = []  # nenhum slot disponível

    basic_node._start_process = mocker.Mock(wraps=basic_node._start_process)

    # Act
    basic_node.process(simulator=mock_simulator)

    # Assert
    mock_queue.skip_process.assert_called_once_with(Process.LOAD)
    mock_queue.recover.assert_called_once()
    basic_node._start_process.assert_not_called()


def test_process_skips_when_blocked(basic_node, mock_train, mock_slot, mock_simulator, mocker):
    # Arrange
    mock_constraint = mocker.Mock()
    mock_constraint.process_type.return_value = Process.LOAD
    mock_constraint.is_blocked.return_value = True
    basic_node.process_constraints = [mock_constraint]
    basic_node.process_units = [mock_slot]

    mock_queue = mocker.Mock()
    mock_queue.running_queue.return_value = [mock_train]
    basic_node.queue_to_enter = mock_queue
    basic_node._start_process = mocker.Mock(wraps=basic_node._start_process)

    # Act
    basic_node.process(simulator=mock_simulator)

    # Assert
    mock_queue.skip_process.assert_called_once_with(Process.LOAD)
    mock_queue.recover.assert_called_once()
    basic_node._start_process.assert_not_called()


def test_dispatch_removes_train_when_not_blocked(basic_node, mock_train, mocker):
    mock_constraint = mocker.Mock()
    mock_constraint.is_blocked.return_value = False
    basic_node.liberation_constraints[mock_train.ID] = [mock_constraint]

    mock_queue = mocker.Mock()
    mock_queue.now.return_value = [mock_train]
    mock_queue.pop.return_value = mock_train
    basic_node.queue_to_leave = mock_queue

    train_picker = []
    mock_train.leave = mocker.Mock()

    basic_node.dispatch(train_picker)

    mock_queue.pop.assert_called_once_with(current_time="12:00")
    mock_train.leave.assert_called_once_with(node=basic_node)
    assert mock_train.ID not in basic_node.liberation_constraints
    assert train_picker == [mock_train]


def test_dispatch_does_nothing_when_blocked(basic_node, mock_train, mocker):
    mock_constraint = mocker.Mock()
    mock_constraint.is_blocked.return_value = True
    basic_node.liberation_constraints[mock_train.ID] = [mock_constraint]

    mock_queue = mocker.Mock()
    mock_queue.now.return_value = [mock_train]
    basic_node.queue_to_leave = mock_queue

    mock_train.leave = mocker.Mock()
    train_picker = []

    basic_node.dispatch(train_picker)

    mock_queue.pop.assert_not_called()
    mock_train.leave.assert_not_called()
    assert train_picker == []