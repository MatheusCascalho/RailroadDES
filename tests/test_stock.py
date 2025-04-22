import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from models.clock import Clock
from models.constants import EventName
from models.exceptions import StockException
from models.stock import OwnStock, StockEvent, StockEventPromise, StockHistory

# ============================================
# Testes StockEvent
# ============================================

def test_stock_event_instantiation():
    now = datetime.now()
    event = StockEvent(event=EventName.RECEIVE_VOLUME, instant=now, volume=100)
    assert event.event == EventName.RECEIVE_VOLUME
    assert event.instant == now
    assert event.volume == 100

# ============================================
# Testes StockEventPromise
# ============================================

def test_partial_event_calculation():
    now = datetime.now()
    promise = StockEventPromise(
        event=EventName.RECEIVE_VOLUME,
        promise_date=now,
        completion_date=now + timedelta(hours=10),
        volume=100
    )
    partial = promise.partial_event(
        current_date=now + timedelta(hours=5),
        max_volume=100
    )
    assert partial.volume == pytest.approx(50)


def test_partial_event_max_volume_limit():
    now = datetime.now()
    promise = StockEventPromise(
        event=EventName.RECEIVE_VOLUME,
        promise_date=now,
        completion_date=now + timedelta(hours=10),
        volume=100
    )
    partial = promise.partial_event(
        current_date=now + timedelta(hours=5),
        max_volume=20
    )
    assert partial.volume == 20


def test_is_done_property():
    now = datetime.now()
    promise = StockEventPromise(
        event=EventName.RECEIVE_VOLUME,
        promise_date=now,
        completion_date=now + timedelta(hours=1),
        volume=0.00000001
    )
    assert promise.is_done


def test_is_done_after_partial_event_completion():
    now = datetime.now()
    promise = StockEventPromise(
        event=EventName.RECEIVE_VOLUME,
        promise_date=now,
        completion_date=now + timedelta(hours=2),
        volume=100
    )
    current_date = promise.completion_date
    max_volume = 1000

    _ = promise.partial_event(current_date=current_date, max_volume=max_volume)

    assert promise.is_done


def test_is_not_done_during_partial_event_in_progress():
    now = datetime.now()
    promise = StockEventPromise(
        event=EventName.RECEIVE_VOLUME,
        promise_date=now,
        completion_date=now + timedelta(hours=4),
        volume=100
    )
    current_date = now + timedelta(hours=1)
    max_volume = 1000

    _ = promise.partial_event(current_date=current_date, max_volume=max_volume)

    assert not promise.is_done

# ============================================
# Testes StockHistory
# ============================================

def test_stock_history_events():
    events = [
        StockEvent(event=EventName.RECEIVE_VOLUME, instant=datetime.now(), volume=50)
    ]
    history = StockHistory(events)
    assert history.events == events

# ============================================
# Testes OwnStock
# ============================================

def create_stock(initial_volume=0, capacity=1000):
    clock = Clock(start=datetime(2025, 4, 1), discretization=timedelta(hours=1))
    stock = OwnStock(clock, capacity=capacity, product="steel", initial_volume=initial_volume)
    return stock


def test_receive_volume():
    clock = Clock(start=datetime(2025, 4, 1), discretization=timedelta(hours=1))
    stock = OwnStock(clock, capacity=1000, product="steel", initial_volume=0)

    stock.receive(100)

    assert stock.volume == 100
    assert stock.events[-1].event == EventName.RECEIVE_VOLUME
    assert stock.events[-1].instant == clock.current_time


def test_dispatch_volume():
    clock = Clock(start=datetime(2025, 4, 1), discretization=timedelta(hours=1))
    stock = OwnStock(clock, capacity=1000, product="steel", initial_volume=200)

    stock.dispatch(50)

    assert stock.volume == 150
    assert stock.events[-1].event == EventName.DISPATCH_VOLUME
    assert stock.events[-1].instant == clock.current_time


def test_receive_over_capacity_raises():
    stock = create_stock(initial_volume=900, capacity=1000)
    with pytest.raises(Exception):
        stock.receive(200)


def test_dispatch_over_volume_raises():
    stock = create_stock(initial_volume=100)
    with pytest.raises(Exception):
        stock.dispatch(200)


def test_save_promise():
    stock = create_stock()
    promise = StockEventPromise(
        event=EventName.RECEIVE_VOLUME,
        promise_date=datetime.now(),
        completion_date=datetime.now() + timedelta(hours=1),
        volume=50
    )
    stock.save_promise([promise])
    assert promise in stock.promises


def test_update_promises():
    stock = create_stock(initial_volume=500, capacity=1000)
    clock = stock.clock

    promise = StockEventPromise(
        event=EventName.RECEIVE_VOLUME,
        promise_date=clock.current_time,
        completion_date=clock.current_time + timedelta(hours=2),
        volume=100
    )
    stock.save_promise([promise])

    clock.update()
    clock.update()
    clock.update()

    stock.update_promises()

    assert promise.is_done or promise.volume < 100


def test_get_last_receive_event_success():
    stock = create_stock()
    stock.receive(50)
    event = stock.get_last_receive_event()
    assert event.event == EventName.RECEIVE_VOLUME


def test_get_last_receive_event_no_event():
    stock = create_stock()
    with pytest.raises(Exception):
        stock.get_last_receive_event()


def test_own_stock_str():
    stock = create_stock(initial_volume=250, capacity=500)
    representation = str(stock)
    assert "Own Stock" in representation
    assert "Volume: 250" in representation
