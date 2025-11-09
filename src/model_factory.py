import json
from src.clock import Clock
from datetime import datetime, timedelta
from pytest import fixture
from src.node_data_model import StockNodeData
from src.node_factory import NodeStockFactory
from src.clock import Clock
from datetime import datetime, timedelta

def simple_stock_node_data_factory(simple_stock, simple_process_rates, simple_rates):
    def make(process='load', has_replanisher=True, process_rate=1.2e3):
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
        for rate in data['rates']:
            rate['rate'] = process_rate

        data = StockNodeData(**data)
        return data
    return make

def simple_stock_node_factory(simple_clock):
    node_data = simple_stock_node_data_factory()
    def make(name, process,clock=None, has_replanisher=True, process_rate=1.2e3):
        data=simple_stock_node_data_factory(process, has_replanisher, process_rate=process_rate)
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

def create_simulation_model(
        simple_clock,
        simple_product,
        simple_train
):
    def create(
            sim,
            n_trains,
            train_size = 6e3,
            process_times = [5,6,10,7,15,8.9,30,18,4,6]
    ):
        clk = Clock(
            start=datetime(2025, 4, 1),
            discretization=timedelta(hours=1)
        )

        with open('../artifacts/transit_times.json', 'r') as f:
            data = json.load(f)

        load_points = []
        unload_points = []
        transit_times = []
        demands = []
        p_times = iter(process_times)
        for tt in data:
            d = simple_stock_node_factory(
                tt['load_destination'],
                clock=simple_clock,
                process='unload',
                has_replanisher=False,
                process_rate=train_size/next(p_times),
            )
            unload_points.append(d)
            o = simple_stock_node_factory(
                tt['load_origin'],
                clock=simple_clock,
                process='load',
                has_replanisher=True,
                process_rate=train_size / next(p_times),
            )
            load_points.append(o)
            transit_time = TransitTime(**tt)
            transit_times.append(transit_time)
            demand = Demand(
                flow=Flow(
                    origin=tt['load_origin'],
                    destination=tt['load_destination'],
                    product=simple_product
                ),
                volume=train_size*10
            )
            demands.append(demand)
        mesh = RailroadMesh(
            load_points=tuple(load_points),
            unload_points=tuple(unload_points),
            transit_times=transit_times
        )

        trains = [
            simple_train(
                simple_product,
                simple_clock,
                train_size=train_size,
                simulator=sim
            )

            for _ in range(n_trains)
        ]
        router = RandomRouter(demands=demands)
        model = Railroad(
            mesh=mesh,
            trains=trains,
            demands=demands,
            router=router
        )

        return model
    return create
