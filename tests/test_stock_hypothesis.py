import pytest
from hypothesis import given, strategies as st, assume, example
from datetime import datetime, timedelta

from src.simulation.clock import Clock
from src.domain.constants import EventName
from src.domain.entities.stock import OwnStock, StockEventPromise

# ============================================
# Estratégias auxiliares
# ============================================

stock_capacities = st.floats(min_value=500, max_value=5000)
# st.floats(min_value=0, max_value=2500) = st.floats(min_value=0, max_value=2500)
# st.floats(min_value=1, max_value=1000) = st.floats(min_value=1, max_value=1000)

# ============================================
# Teste de Recebimentos com Hypothesis
# ============================================

@given(
    capacity=st.floats(min_value=500, max_value=5000),
    initial_volume=st.floats(min_value=0, max_value=2500),
    receive_amount=st.floats(min_value=1, max_value=1000),
    alpha=st.floats(min_value=0, max_value=1, exclude_min=True, exclude_max=True)
)
def test_hypothesis_receive(capacity, initial_volume, receive_amount, alpha):
    assume(initial_volume <= capacity)

    clock = Clock(start=datetime(2025, 4, 1), discretization=timedelta(hours=1))
    stock = OwnStock(clock, capacity=capacity, product='hypothesis_product', initial_volume=initial_volume)
    r1 = alpha * receive_amount
    r2 = (1-alpha) * receive_amount

    if stock.volume + receive_amount <= stock.capacity:
        stock.receive(r1)
        clock.jump(timedelta(2))
        stock.receive(r2)
        assert stock.capacity == pytest.approx(stock.space + stock.volume), f"Volume + estoque != capacidade"
        assert stock.volume == pytest.approx(receive_amount + initial_volume), "Volume inicial + recebido != volume atual"
        assert 0 <= stock.volume <= stock.capacity, "Volume fora do range do estoque"
        assert stock.get_last_receive_event().volume == r2, "Último evento registrado com valor errado"
        assert len(stock.history().events) == 2 if initial_volume == 0 else 3, "Quantidade de eventos registrados incorreta!"

# ============================================
# Teste de Despachos com Hypothesis
# ============================================

@given(
    capacity=st.floats(min_value=500, max_value=5000),
    initial_volume=st.floats(min_value=0, max_value=2500),
    dispatch_amount=st.floats(min_value=1, max_value=1000)
)
def test_hypothesis_dispatch(capacity, initial_volume, dispatch_amount):
    assume(initial_volume <= capacity)

    clock = Clock(start=datetime(2025, 4, 1), discretization=timedelta(hours=1))
    stock = OwnStock(clock, capacity=capacity, product='hypothesis_product', initial_volume=initial_volume)

    if stock.volume - dispatch_amount >= 0:
        stock.dispatch(dispatch_amount)
        assert stock.capacity == pytest.approx(stock.space + stock.volume)
        assert 0 <= stock.volume <= stock.capacity

# ============================================
# Teste de Promessas com Hypothesis
# ============================================

@given(
    capacity=st.floats(min_value=500, max_value=5000),
    initial_volume=st.floats(min_value=0, max_value=2500),
    volume_promise=st.floats(min_value=1, max_value=1000)
)
# @example(capacity=500.0, initial_volume=1.0, receive_amount=1000.0)
def test_hypothesis_promises(capacity, initial_volume, volume_promise):
    assume(initial_volume <= capacity)
    assume(volume_promise <= capacity - initial_volume)

    clock = Clock(start=datetime(2025, 4, 1), discretization=timedelta(hours=1))
    stock = OwnStock(clock, capacity=capacity, product='hypothesis_product', initial_volume=initial_volume)

    promise_date = clock.current_time
    completion_date = clock.current_time + timedelta(hours=2)

    promise = StockEventPromise(
        event=EventName.RECEIVE_VOLUME,
        promise_date=promise_date,
        completion_date=completion_date,
        volume=volume_promise
    )

    stock.save_promise([promise])

    for _ in range(5):
        clock.update()
        stock.update_promises()

    # Promessa deve estar consumida ou dentro do limite
    assert promise.volume >= 0
    assert promise.volume <= stock.capacity
