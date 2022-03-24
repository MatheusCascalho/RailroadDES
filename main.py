from models.des_simulator import DESSimulator
from models.des_model import Railroad
from models.conditions import RailroadMesh, TransitTime
from models.node import Node
from models.train import Train
from datetime import timedelta, datetime


load_points = (
    Node(
        queue_capacity=3,
        name='Terminal',
        slots=3
    ),
)

unload_points = (
    Node(
        queue_capacity=3,
        name='Port',
        slots=3
    ),
)

transit_times = [
    TransitTime(
        load_origin=load_points[0].name,
        load_destination=unload_points[0].name,
        empty_time=timedelta(hours=4),
        loaded_time=timedelta(hours=5)
    ),
]

mesh = RailroadMesh(
    load_points=load_points,
    unload_points=unload_points,
    transit_times=transit_times
)

trains = [
    Train(
        id=0,
        origin=0,
        destination=1,
        model=0,
        path=[0, 1, 0],
        capacity=1000.0
    )
]
model = Railroad(mesh=mesh, trains=trains, demands=None)
time_horizon = timedelta(days=5)
simulator = DESSimulator(initial_date=datetime(2020, 1, 1))
simulator.simulate(model=model, time_horizon=time_horizon)
print("Fim")
