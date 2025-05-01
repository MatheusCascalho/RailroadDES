from pytest import fixture
from models.demand import Flow, Demand
from models.task import Task
from models.train import Train

@fixture
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
            is_loaded=False,
            clock=clk
        )
        return train

    return make
