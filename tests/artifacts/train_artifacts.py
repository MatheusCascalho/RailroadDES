from datetime import timedelta

from pytest import fixture
from models.demand import Flow, Demand
from models.railroad import RailSegment
from models.task import Task
from models.train import Train
from models.arrive_scheduler import ArriveScheduler
from tests.test_integration import FakeSimulator


@fixture
def simple_train(basic_node_factory):
    def make(product, clk, simulator, train_size=6e3):
        flow = Flow(
            origin="origin",
            destination="destination",
            product=product
        )
        demand = Demand(
            flow=flow,
            volume=103e3
        )
        segment1 = RailSegment(
            origin=basic_node_factory("origin"),
            destination=basic_node_factory("destination"),
            time_to_origin=timedelta(hours=10),
            time_to_destination=timedelta(hours=10)
        )
        segment2 = segment1.reversed()
        scheduler = ArriveScheduler(
            rail_segments=[segment1, segment2],
            simulator=simulator
        )
        task = Task(
            demand=demand,
            path=["origin", "destination"],
            task_volume=train_size,
            current_time=clk.current_time,
            scheduler=scheduler
        )
        train = Train(
            capacity=train_size,
            task=task,
            is_loaded=False,
            clock=clk
        )
        return train

    return make
