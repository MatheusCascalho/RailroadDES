import pytest
from datetime import datetime, timedelta
from models.task import Task, TimeEvent, Process
from models.constants import EventName
from models.demand import Demand, Flow
from models.exceptions import EventSequenceError


# Fixture para criar uma instância de Demand (pode ser substituída por uma classe mock)
@pytest.fixture
def demand():
    flow = Flow(origin='origen', destination='destination', product='product')
    return Demand(flow=flow, volume=10e3)  # Supondo que Demand seja uma classe existente ou um mock.


@pytest.fixture
def task(demand):
    """Fixture para criar uma instância de Task com valores padrões."""
    current_time = datetime(2025, 4, 10, 12, 0)
    task_volume = 100.0  # Defina o volume da tarefa
    return Task(
        demand=demand,
        task_volume=task_volume,
        current_time=current_time,
        path=[],
        state={}
    )


def test_task_initialization(task):
    """Testa a inicialização da tarefa e seus atributos."""
    assert task.ID.startswith("task_")  # Verifica se o ID é gerado corretamente
    assert task.demand is not None  # Verifica se a demanda está associada à tarefa
    assert task.task_volume == 100.0  # Verifica se o volume da tarefa está correto
    assert task.invoiced_volume == 0  # Verifica se o volume faturado inicial é 0
    assert task.time_table is not None  # Verifica se o TimeTable foi criado corretamente


def test_update_with_finish_process(task):
    """Testa a atualização com o evento FINISH_PROCESS e verifica o cálculo do volume faturado."""
    event1 = TimeEvent(event=EventName.ARRIVE, instant=datetime(2025, 4, 10, 12, 0))
    event2 = TimeEvent(event=EventName.START_PROCESS, instant=datetime(2025, 4, 10, 12, 15))
    event3 = TimeEvent(event=EventName.FINISH_PROCESS, instant=datetime(2025, 4, 10, 12, 45))

    # Atualiza a tarefa com os eventos
    task.update(event1, Process.LOAD)  # ARRIVE
    task.update(event2, Process.LOAD)  # START_PROCESS
    task.update(event3, Process.LOAD)  # FINISH_PROCESS

    # Verifica se o volume faturado é atualizado corretamente
    assert task.invoiced_volume == 100.0  # Volume faturado deve ser igual ao volume da tarefa


def test_update_without_finish_process(task):
    """Testa que o volume faturado não é atualizado se FINISH_PROCESS não foi registrado."""
    event1 = TimeEvent(event=EventName.ARRIVE, instant=datetime(2025, 4, 10, 12, 0))
    event2 = TimeEvent(event=EventName.START_PROCESS, instant=datetime(2025, 4, 10, 12, 15))

    # Atualiza a tarefa com ARRIVE e START_PROCESS, mas não FINISH_PROCESS
    task.update(event1, Process.LOAD)
    task.update(event2, Process.LOAD)

    # Verifica se o volume faturado não foi atualizado
    assert task.invoiced_volume == 0  # O volume faturado não deve ser alterado sem FINISH_PROCESS


def test_penalty(task):
    """Testa o cálculo da penalidade baseado no tempo de fila (queue_time)."""
    event1 = TimeEvent(event=EventName.ARRIVE, instant=datetime(2025, 4, 10, 12, 0))
    event2 = TimeEvent(event=EventName.START_PROCESS, instant=datetime(2025, 4, 10, 12, 15))
    event3 = TimeEvent(event=EventName.FINISH_PROCESS, instant=datetime(2025, 4, 10, 12, 45))
    event4 = TimeEvent(event=EventName.DEPARTURE, instant=datetime(2025, 4, 10, 13, 0))

    task.update(event1, Process.LOAD)
    task.update(event2, Process.LOAD)
    task.update(event3, Process.LOAD)
    task.update(event4, Process.LOAD)

    # Verifica o cálculo da penalidade, que depende do tempo de fila
    queue_to_enter = 15
    queue_to_leave = 15
    queue = queue_to_enter + queue_to_leave
    assert task.penalty().total_seconds() == timedelta(minutes=queue).total_seconds()  # Tempo de fila de 15 minutos (de ARRIVE a START_PROCESS)


def test_reward(task):
    """Testa o cálculo da recompensa baseado no volume faturado (invoiced_volume)."""
    event1 = TimeEvent(event=EventName.ARRIVE, instant=datetime(2025, 4, 10, 12, 0))
    event2 = TimeEvent(event=EventName.START_PROCESS, instant=datetime(2025, 4, 10, 12, 15))
    event3 = TimeEvent(event=EventName.FINISH_PROCESS, instant=datetime(2025, 4, 10, 12, 45))
    event4 = TimeEvent(event=EventName.DEPARTURE, instant=datetime(2025, 4, 10, 13, 0))

    task.update(event1, Process.LOAD)
    task.update(event2, Process.LOAD)
    task.update(event3, Process.LOAD)
    task.update(event4, Process.LOAD)

    # Verifica o cálculo da recompensa, que é o volume faturado
    assert task.reward() == 100.0  # A recompensa é o volume faturado, que é igual ao volume da tarefa


def test_update_with_invalid_event_sequence(task):
    """Testa que uma exceção é levantada ao tentar atualizar a tarefa com um evento fora de sequência (DEPARTURE antes de FINISH_PROCESS)."""
    event1 = TimeEvent(event=EventName.ARRIVE, instant=datetime(2025, 4, 10, 12, 0))
    event2 = TimeEvent(event=EventName.DEPARTURE,
                       instant=datetime(2025, 4, 10, 13, 0))  # Tentando registrar DEPARTURE antes de FINISH_PROCESS

    task.update(event1, Process.LOAD)

    with pytest.raises(EventSequenceError):
        task.update(event2, Process.LOAD)

