from pytest import fixture
from src.node_data_model import StockNodeData
from src.node_factory import NodeStockFactory
from src.clock import Clock
from datetime import datetime, timedelta

@fixture
def simple_product():
    return "product"

@fixture
def simple_stock(simple_product):
    return [{"product": simple_product, "capacity": 60e3, "initial_volume": 5}]

@fixture
def simple_stock_factory(simple_product):
    def make(initial_volume=5):
        return [{"product": simple_product, "capacity": 60e3, "initial_volume": initial_volume}]
    return make



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
        "queue_capacity": 20,
        "post_operation_time": 5
    }
    data = StockNodeData(**data)
    return data

@fixture
def simple_stock_node_data_factory(simple_stock_factory, simple_process_rates, simple_rates):
    def make(process='load', has_replanisher=True, process_rate=1.2e3, initial_stock=5):
        simple_process_rates[0]['type']=process
        data = {
            "name": "xpto",
            "stocks": simple_stock_factory(initial_volume=initial_stock),
            "rates": simple_process_rates,
            "replenishment": simple_rates if has_replanisher else [],
            "train_sizes": [6e3],
            "queue_capacity": 20,
            "post_operation_time": 5
        }
        for rate in data['rates']:
            rate['rate'] = process_rate

        data = StockNodeData(**data)
        return data
    return make

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

@fixture
def simple_stock_node_factory(simple_clock, simple_stock_node_data_factory):
    def make(name, process,clock=None, has_replanisher=True, process_rate=1.2e3, initial_stock=5):
        data=simple_stock_node_data_factory(process, has_replanisher, process_rate=process_rate, initial_stock=initial_stock)
        data.name = name
        if data.replenishment:
            data.replenishment[0].rate = process_rate / 10
        if clock:
            clk = clock
        else:
            clk = simple_clock
        factory = NodeStockFactory(
            clock=clk,
            data=data
        )
        node = factory.create_node()
        return node
    return make




