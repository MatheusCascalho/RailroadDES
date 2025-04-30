from itertools import product

import pytest

from models.des_simulator import DESSimulator
from models.node import Node, StockNode
from models.node_constraints import ProcessConstraintSystem
from models.stock_constraints import StockToLoadTrainConstraint
from datetime import timedelta, datetime
from models.clock import Clock
from models.train import Train
from models.task import Task
from models.demand import Flow, Demand
from models.stock import OwnStock
from models.stock_replanish import SimpleStockReplanisher, ReplenishRate
from models.data_model import ProcessorRate
from models.constants import Process


# @pytest.fixture
class FakeSimulator(DESSimulator):
    def __init__(self, clock: Clock):
        super().__init__(initial_date=datetime(2025,4,1), clock=clock)


@pytest.fixture
def simple_train():
    def make(product, clk, train_size=6e3):
        flow = Flow(
            origin="origin",
            destination="destination",
            product=product
        )
        demand = Demand(
            flow=flow,
            volume=103e3
        )
        task = Task(
            demand=demand,
            path=["origin", "destination"],
            task_volume=train_size,
            current_time=clk.current_time
        )
        train = Train(
            capacity=train_size,
            task=task,
            is_loaded=False
        )
        return train

    return make

def test_simulation(simple_train):
    clk = Clock(
        start=datetime(2025,4,1),
        discretization=timedelta(hours=1)
    )
    sim = FakeSimulator(clock=clk)
    product = "product"
    train = simple_train(product=product, clk=clk)
    constraint = ProcessConstraintSystem()
    process_time = timedelta(hours=5)

    rates = {
        product: [ProcessorRate(product=product,type=Process.LOAD,rate=40)]
    }
    node = Node(
        queue_capacity=20,
        name="xpto",
        clock=clk,
        load_constraint_system=constraint,
        unload_constraint_system=constraint,
        process_rates=rates
    )


    # Simulacao
    train.arrive(simulator=sim, node=node)
    node.receive(train)
    node.process(simulator=sim)
    train.start_load(simulator=sim, process_time=process_time)
    clk.jump(process_time)
    train.finish_load(simulator=sim, node=node)
    node.maneuver_to_dispatch(simulator=sim)
    assert True

def test_stock_node_simulation(simple_train, simple_stock_node, simple_clock):

    sim = FakeSimulator(clock=simple_clock)
    product = "product"
    train_size = 6e3
    train = simple_train(product, simple_clock, train_size)

    node = simple_stock_node
    clk = simple_clock
    process_time = timedelta(hours=5)

    # Simulacao
    train.arrive(simulator=sim, node=node)
    node.receive(train)
    node.process(simulator=sim)
    try:
        train.start_load(simulator=sim, process_time=process_time)
        clk.jump(process_time)
        train.finish_load(simulator=sim, node=node)
        node.maneuver_to_dispatch(simulator=sim)
    except Exception as e:
        clk.jump(timedelta(10))
        node.process(simulator=sim)

        train.start_load(simulator=sim, process_time=process_time)
        clk.jump(timedelta(hours=3))
        node.process(simulator=sim)

        clk.jump(timedelta(hours=2))
        train.finish_load(simulator=sim, node=node)
        node.maneuver_to_dispatch(simulator=sim)


    assert True