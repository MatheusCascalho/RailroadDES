import pytest
from datetime import datetime, timedelta
from src.simulation.clock import Clock
from src.domain.entities.stock import OwnStock
from src.events.stock_replanish import SimpleStockReplanisher, ReplenishRate
from src.domain.exceptions import StockException


@pytest.fixture
def setup_clock_and_stock():
    clk = Clock(start=datetime(2025, 4, 1), discretization=timedelta(hours=1))
    stock = OwnStock(clock=clk, capacity=50, product='aço')
    return clk, stock


@pytest.fixture
def setup_replenisher(setup_clock_and_stock):
    clk, stock = setup_clock_and_stock
    replenish_rates = [
        ReplenishRate(product='aço', rate=5.0),  # 5 unidades de aço por hora
        ReplenishRate(product='carvão', rate=3.0)  # 3 unidades de carvão por hora
    ]
    replenisher = SimpleStockReplanisher(replenish_rates=replenish_rates, clock=clk)
    return replenisher, stock


def test_initial_stock_volume(setup_clock_and_stock):
    # Testando o volume inicial do estoque
    _, stock = setup_clock_and_stock
    assert stock.volume == 0  # Inicialmente, o volume do estoque é 0


def test_replenish_volume_after_one_hour(setup_replenisher):
    # Testando o reabastecimento do estoque
    replanisher, stock = setup_replenisher
    stock.clock.update()
    # Realizando o reabastecimento
    replanisher.replenish([stock])

    # Verificando o volume após reabastecimento
    assert stock.volume == 5  # O volume do estoque deve ter aumentado


def test_replenish_volume_after_one_hour_and_stock_already_receive_an_event(setup_replenisher):
    # Testando o reabastecimento do estoque
    replanisher, stock = setup_replenisher
    stock.receive(volume=10)
    stock.clock.update()
    # Realizando o reabastecimento
    replanisher.replenish([stock])

    # Verificando o volume após reabastecimento
    assert stock.volume == 15  # O volume do estoque deve ter aumentado
    assert len(stock.history().events) == 2



def test_replenish_volume_after_five_hours(setup_replenisher):
    # Testando o reabastecimento do estoque
    replanisher, stock = setup_replenisher
    stock.clock.jump(timedelta(hours=5))
    # Realizando o reabastecimento
    replanisher.replenish([stock])
    rate = 5
    time = 5
    expected_volume = rate * time
    # Verificando o volume após reabastecimento
    assert stock.volume == expected_volume  # O volume do estoque deve ter aumentado


def test_replenish_with_capacity_limit(setup_replenisher):
    # Testando o reabastecimento respeitando o limite de capacidade do estoque
    replanisher, stock = setup_replenisher

    # Avançando o tempo para reabastecer mais do que a capacidade do estoque
    stock.clock.jump(timedelta(hours=15))  # 15 horas
    replanisher.replenish([stock])

    # O volume não pode exceder a capacidade do estoque
    assert stock.volume == stock.capacity  # O volume deve ser igual à capacidade do estoque


def test_replenish_for_multiple_products(setup_clock_and_stock):
    # Testando o reabastecimento para múltiplos produtos
    clk, stock = setup_clock_and_stock
    replenish_rates = [
        ReplenishRate(product='aço', rate=5.0),
        ReplenishRate(product='carvão', rate=3.0)
    ]
    replanisher = SimpleStockReplanisher(replenish_rates=replenish_rates, clock=clk)
    clk.update()

    # Criando dois estoques diferentes
    stock_aco = OwnStock(clock=clk, capacity=100, product='aço')
    stock_carvao = OwnStock(clock=clk, capacity=100, product='carvão')

    # Reabastecendo os dois estoques
    replanisher.replenish([stock_aco, stock_carvao])

    # Verificando os volumes após reabastecimento
    assert stock_aco.volume == 5.0  # Aço deve ter 5 unidades após 1 hora
    assert stock_carvao.volume == 3.0  # Carvão deve ter 3 unidades após 1 hora


def test_replenish_with_missing_rate(setup_clock_and_stock):
    # Testando o comportamento quando a taxa de reposição para um produto não está definida
    clk, stock = setup_clock_and_stock
    replenish_rates = [
        ReplenishRate(product='aço', rate=5.0)
    ]
    replanisher = SimpleStockReplanisher(replenish_rates=replenish_rates, clock=clk)

    # Estoque de 'carvão' não tem taxa de reposição definida
    stock_carvao = OwnStock(clock=clk, capacity=100, product='carvão')

    # O reabastecimento para 'carvão' não deve ser feito, já que não existe taxa
    with pytest.raises(Exception, match="Taxa do produto carvão não foi cadastrada!!"):
        replanisher.replenish([stock_carvao])
