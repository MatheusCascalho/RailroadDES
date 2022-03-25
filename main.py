from models.des_simulator import DESSimulator
from models.des_model import Railroad
from models.conditions import RailroadMesh, TransitTime
from models.node import Node
from models.train import Train
from models.demand import Demand
from datetime import timedelta, datetime


load_points = (
    Node(
        queue_capacity=3,
        name='Terminal',
        slots=3,
        process_time=timedelta(hours=7)
    ),
)

unload_points = (
    Node(
        queue_capacity=3,
        name='Port1',
        slots=3,
        process_time=timedelta(hours=6)
    ),
    Node(
        queue_capacity=3,
        name='Port2',
        slots=3,
        process_time=timedelta(hours=10)
    ),
)

transit_times = [
    TransitTime(
        load_origin=load_points[0].name,
        load_destination=unload_points[0].name,
        empty_time=timedelta(hours=17),
        loaded_time=timedelta(hours=20)
    ),
    TransitTime(
        load_origin=load_points[0].name,
        load_destination=unload_points[1].name,
        empty_time=timedelta(hours=17),
        loaded_time=timedelta(hours=20)
    ),
]

mesh = RailroadMesh(
    load_points=load_points,
    unload_points=unload_points,
    transit_times=transit_times
)

trains = [
    Train(
        id=train_id,
        origin=0,
        destination=1,
        model=1,
        path=[],
        current_location=1,
        capacity=1000.0
    )
    for train_id in range(5)
]
demands = [
    Demand(
        origin='Terminal',
        destination='Port1',
        volume=14e3,
    ),
    Demand(
        origin='Terminal',
        destination='Port2',
        volume=3e3,
    ),
]

model = Railroad(mesh=mesh, trains=trains, demands=demands)
time_horizon = timedelta(days=15)
simulator = DESSimulator(initial_date=datetime(2020, 1, 1))
simulator.simulate(model=model, time_horizon=time_horizon)
print("Fim")
