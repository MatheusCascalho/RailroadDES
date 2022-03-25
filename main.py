from models.des_simulator import DESSimulator
from models.des_model import Railroad
from models.conditions import RailroadMesh, TransitTime
from models.node import Node
from models.train import Train
from models.demand import Demand
from datetime import timedelta, datetime
import pandas as pd
pd.set_option('display.max_columns', None)

load_points = (
    Node(
        queue_capacity=3,
        name='Terminal',
        slots=1,
        process_time=timedelta(hours=7)
    ),
)

unload_points = (
    Node(
        queue_capacity=3,
        name='Port1',
        slots=1,
        process_time=timedelta(hours=6)
    ),
    Node(
        queue_capacity=3,
        name='Port2',
        slots=1,
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
    for train_id in range(2)
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
print("="*40 + "Início da simulação" + "="*40)
simulator.simulate(model=model, time_horizon=time_horizon)
print("="*40 + "Fim da simulação" + "="*40)
print("="*40 + "Estatísticas" + "="*40)
print("-"*20 + "Volume Operado" + "-"*20)
print(model.statistics())
print("-"*20 + "Tabela de tempos" + "-"*20)
for train in trains:
    for node, table in train.time_table.items():
        print("-" * 20 + mesh.id_to_name[node] + "-" * 20)
        print(pd.DataFrame(table))

# for train in trains:
#

