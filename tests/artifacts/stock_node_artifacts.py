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
        "queue_capacity": 20,
        "post_operation_time": 5
    }
    data = StockNodeData(**data)
    return data

@fixture
def simple_stock_node_data_factory(simple_stock, simple_process_rates, simple_rates):
    def make(process='load', has_replanisher=True):
        simple_process_rates[0]['type']=process
        data = {
            "name": "xpto",
            "stocks": simple_stock,
            "rates": simple_process_rates,
            "replenishment": simple_rates if has_replanisher else [],
            "train_sizes": [6e3],
            "queue_capacity": 20,
            "post_operation_time": 5
        }

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
    def make(name, process,clock=None, has_replanisher=True):
        data=simple_stock_node_data_factory(process, has_replanisher)
        data.name = name
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




