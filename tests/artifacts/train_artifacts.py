from datetime import timedelta

from pytest import fixture
from models.demand import Flow, Demand
from models.railroad_mesh import RailSegment
from models.task import Task
from models.train import Train
from models.arrive_scheduler import ArriveScheduler


@fixture
def simple_train(basic_node_factory, simple_clock, simple_simulator):
    def make(product='product', clk=simple_clock, simulator=simple_simulator, train_size=6e3, add_scheduler=False):
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
            destination=basic_node_factory("origin"),
            origin=basic_node_factory("destination"),
            time_to_origin=timedelta(hours=10),
            time_to_destination=timedelta(hours=10)
        )
        segment2 = segment1.reversed()

        task = Task(
            demand=demand,
            path=["origin", "destination"],
            task_volume=train_size,
            current_time=clk.current_time,
            state=""
        )
        train = Train(
            capacity=train_size,
            task=task,
            is_loaded=False,
            clock=clk
        )
        if add_scheduler:
            scheduler = ArriveScheduler(
                rail_segments=[segment1, segment2],
                simulator=simulator
            )
            train.add_observers([scheduler])
        return train

    return make
