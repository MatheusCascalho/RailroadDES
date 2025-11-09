from src.des_simulator import DESSimulator
from src.domain.entities.railroad import Railroad
from src.domain.entities.railroad_mesh import RailroadMesh, TransitTime
from src.domain.entities.node import Node
from src.domain.entities.train import Train
from src.domain.entities.demand import Demand
from datetime import timedelta, datetime
import pandas as pd
pd.set_option('display.max_columns', None)


def create_model(demand=[3500, 0], n_trains=1, terminal_times = [7], port_times=[6, 10], queue_capacity=50):
    load_points = (
        Node(
            queue_capacity=queue_capacity,
            name='Terminal',
            slots=1,
            process_time=timedelta(hours=terminal_times[0])
        ),
    )

    unload_points = (
        Node(
            queue_capacity=queue_capacity,
            name='Port1',
            slots=1,
            process_time=timedelta(hours=port_times[0])
        ),
        Node(
            queue_capacity=queue_capacity,
            name='Port2',
            slots=1,
            process_time=timedelta(hours=port_times[1])
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
        for train_id in range(n_trains)
    ]
    demands = [
        Demand(
            origin='Terminal',
            destination='Port1',
            volume=demand[0],
        ),
        Demand(
            origin='Terminal',
            destination='Port2',
            volume=demand[1],
        ),
    ]

    model = Railroad(mesh=mesh, trains=trains, demands=demands)
    return model


def statistics(model):
    print("="*40 + "Estatísticas" + "="*40)
    print("-"*20 + "Volume Operado" + "-"*20)
    print(model.statistics())
    # print("-"*20 + "Tabela de tempos" + "-"*20 + "\n")
    #
    # timetables = []
    # for train in model.trains:
    #     for node, registers in train.time_table.items():
    #         df = pd.DataFrame(registers)
    #         df['train'] = f"TREM {train.id}"
    #
    #         processing = df[['start_process','finish_process', 'train']]
    #         processing['node'] = model.mesh.node_by_id(node).name
    #         processing.rename(columns={"start_process": "start", "finish_process": "end"}, inplace=True)
    #
    #         queue_to_enter = df[['arrive','start_process', 'train']]
    #         queue_to_enter['node'] = f"fila de entrada em {model.mesh.node_by_id(node).name}"
    #         queue_to_enter.rename(columns={"arrive": "start", "start_process": "end"}, inplace=True)
    #
    #         queue_to_leave = df[['finish_process','departure', 'train']]
    #         queue_to_leave['node'] = f"fila de saída em {model.mesh.node_by_id(node).name}"
    #         queue_to_leave.rename(columns={"finish_process": "start", "departure": "end"},inplace=True)
    #
    #         timetables.extend([processing, queue_to_enter, queue_to_leave])
    # df = pd.concat(timetables)
    # df


    # fig = px.timeline(df, x_start="start", x_end="end", y="node", color="train")
    # fig.update_yaxes(autorange="reversed")
    # fig.write_html('simulation.md.html')
    # display(HTML(filename='simulation.md.html'))

model = create_model(demand=[14000, 14000], n_trains=4, terminal_times = [7], port_times=[6, 10], queue_capacity=3)
time_horizon = timedelta(days=2)
simulator = DESSimulator(initial_date=datetime(2020, 1, 1))
print("="*40 + "Início da simulação" + "="*40)
simulator.simulate(model=model, time_horizon=time_horizon)
print("="*40 + "Fim da simulação" + "="*40)
statistics(model)