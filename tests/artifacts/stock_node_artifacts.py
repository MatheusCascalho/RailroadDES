from pytest import fixture
from models.data_model import StockNodeData
from models.node_factory import NodeStockFactory
from models.clock import Clock
from datetime import datetime, timedelta

@fixture
def simple_product():
    return "product"

@fixture
def simple_stock(simple_product):
    return [{"product": simple_product, "capacity": 60e3, "initial_volume": 5}]

@fixture
def simple_process_rates(simple_product):
    return [dict(product=simple_product, type='load', rate=1.2e3)]

@fixture
def simple_rates(simple_product):
    return [dict(product=simple_product, rate=1e3)]

@fixture
def simple_stock_node_data(simple_stock, simple_process_rates, simple_rates):
    data = {
        "name": "xpto",
        "stocks": simple_stock,
        "rates": simple_process_rates,
        "replenishment": simple_rates,
        "train_sizes": [6e3],
        "queue_capacity": 20
    }
    data = StockNodeData(**data)
    return data

@fixture
def simple_clock():
    clk = Clock(
        start=datetime(2025, 4, 1),
        discretization=timedelta(hours=1)
    )
    return clk

@fixture
def simple_stock_node(simple_clock, simple_stock_node_data):
    factory = NodeStockFactory(
        clock=simple_clock,
        data=simple_stock_node_data
    )
    node = factory.create_node()
    return node