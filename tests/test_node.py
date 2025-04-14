import pytest

from models.des_simulator import DESSimulator
from models.node import Node
from models.discrete_event_system import ProcessConstraintSystem
from datetime import timedelta, datetime
from models.clock import Clock
from models.train import Train
from models.task import Task
from models.demand import Flow, Demand

# @pytest.fixture
class FakeSimulator(DESSimulator):
    def __init__(self, clock: Clock):
        super().__init__(initial_date=datetime(2025,4,1), clock=clock)



def test_simulation():
    clk = Clock(
        start=datetime(2025,4,1),
        discretization=timedelta(hours=1)
    )
    sim = FakeSimulator(clock=clk)

    flow = Flow(
        origin="origin",
        destination="destination",
        product="product"
    )
    demand = Demand(
        flow=flow,
        volume=103e3
    )
    task = Task(
        demand=demand,
        path=["origin", "destination"],
        task_volume=6e3,
        current_time=clk.current_time
    )
    train = Train(
        capacity=6e3,
        task=task,
        is_loaded=False
    )
    constraint = ProcessConstraintSystem()
    process_time = timedelta(hours=5)
    node = Node(
        queue_capacity=20,
        name="xpto",
        process_time=process_time,
        clock=clk,
        load_constraint_system=constraint,
        unload_constraint_system=constraint,
        load_units_amount=1,
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