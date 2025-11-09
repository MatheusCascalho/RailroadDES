import pytest
from src.stock import OwnStock
from src.clock import Clock
from datetime import datetime, timedelta
from src.stock_constraints import StockToLoadTrainConstraint, StockToUnloadTrainConstraint


@pytest.fixture
def setup_clock_and_stock():
    clk = Clock(start=datetime(2025, 4, 1), discretization=timedelta(hours=1))
    stock = OwnStock(clock=clk, capacity=50, product='aço')
    return clk, stock

@pytest.fixture
def setup_restrictions():
    restriction_1 = StockToLoadTrainConstraint(train_size=10)
    restriction_2 = StockToLoadTrainConstraint(train_size=5)
    return restriction_1, restriction_2

def test_state_machine_initialization(setup_clock_and_stock, setup_restrictions):
    # Configurando o estoque e as restrições
    _, stock = setup_clock_and_stock
    restriction_1, _ = setup_restrictions

    # Adicionando as restrições como observadores
    stock.add_observers([restriction_1])

    # Verificando que o estado inicial é "READY"
    assert restriction_1.is_blocked() is True  # Inicialmente, bloqueado, pois o volume é 0

def test_state_transition_block_and_release(setup_clock_and_stock, setup_restrictions):
    # Configurando o estoque e as restrições
    _, stock = setup_clock_and_stock
    restriction_1, restriction_2 = setup_restrictions

    # Adicionando as restrições como observadores
    stock.add_observers([restriction_1, restriction_2])

    # Verificando o estado inicial (bloqueado)
    assert restriction_1.is_blocked() is True
    assert restriction_2.is_blocked() is True

    # Modificando o volume no estoque
    stock.receive(volume=15)

    # Verificando se o estado foi liberado
    assert not restriction_1.is_blocked()  # Volume suficiente para liberar a restrição
    assert not restriction_2.is_blocked()  # Volume suficiente para liberar a restrição

    # Despachando carga
    stock.dispatch(volume=10)

    # Verificando os estados após despacho
    assert restriction_1.is_blocked() is True  # Restrição 1 deve voltar a ser bloqueada
    assert not restriction_2.is_blocked()  # Restrição 2 ainda deve estar liberada

def test_stock_load_constraint(setup_clock_and_stock, setup_restrictions):
    # Configurando o estoque e as restrições
    _, stock = setup_clock_and_stock
    restriction_1, _ = setup_restrictions

    # Adicionando a restrição como observador
    stock.add_observers([restriction_1])

    # Verificando o estado inicial
    assert restriction_1.is_blocked() is True  # Inicialmente bloqueado devido ao volume

    # Modificando o volume no estoque
    stock.receive(volume=15)

    # Verificando se o estado foi liberado
    assert not restriction_1.is_blocked()

def test_stock_unload_constraint(setup_clock_and_stock):
    # Testando o comportamento da restrição de espaço (StockUnloadConstraint)
    _, stock = setup_clock_and_stock
    restriction = StockToUnloadTrainConstraint(train_size=10)

    # Adicionando a restrição como observador
    stock.add_observers([restriction])
    stock.receive(volume=50)
    # Verificando o estado inicial
    assert restriction.is_blocked() is True  # Inicialmente bloqueado devido a falta de espaço

    # Modificando o espaço no estoque
    stock.dispatch(volume=15)

    # Verificando se o estado foi liberado
    assert not restriction.is_blocked()

def test_observer_pattern(setup_clock_and_stock):
    # Testando o padrão de observador
    _, stock = setup_clock_and_stock
    restriction_1 = StockToLoadTrainConstraint(train_size=10)
    restriction_2 = StockToLoadTrainConstraint(train_size=5)

    # Adicionando as restrições como observadores
    stock.add_observers([restriction_1, restriction_2])

    # Verificando os estados antes de modificar o estoque
    assert restriction_1.is_blocked() is True
    assert restriction_2.is_blocked() is True

    # Modificando o volume no estoque
    stock.receive(volume=20)

    # Verificando se os observadores foram atualizados corretamente
    assert not restriction_1.is_blocked()
    assert not restriction_2.is_blocked()

    # Despachando carga
    stock.dispatch(volume=15)

    # Verificando os estados após o despacho
    assert restriction_1.is_blocked() is True
    assert not restriction_2.is_blocked()
