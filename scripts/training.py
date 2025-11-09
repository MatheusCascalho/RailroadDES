import sys

# Adicionando o diretório ao sys.path
sys.path.append('../')

import json
from src.control.router import RepeatedRouter, ChainedHistoryRouter
from src.domain.entities.demand import Flow
from src.statistics.stock_graphic import StockGraphic
from tests.artifacts.railroad_artifacts import create_model
from src.des_simulator import DESSimulator
from src.statistics.gantt import Gantt
from src.statistics.operated_volume import OperatedVolume
from datetime import timedelta
from src.simulation.clock import Clock
from datetime import datetime
import dill
from tqdm import tqdm
from time import sleep


def simulate(to_repeat: bool):
    sim = DESSimulator(clock=Clock(start=datetime(2025,4,1), discretization=timedelta(hours=1)))
    # model = create_model(sim=sim, n_trains=3)
    with open('../tests/artifacts/model_2.dill', 'rb') as f:
        model = dill.load(f)
        model.router = ChainedHistoryRouter(model.router.demands)
        sim.clock = model.mesh.load_points[0].clock

    if to_repeat:
        with open('decisions.json', 'r') as f:
            decisions = json.load(f)
        model.router = RepeatedRouter(model.router.demands, to_repeat=[Flow(**d) for d in decisions])
    sim.simulate(model=model, time_horizon=timedelta(days=20))

    return model

try:
    with open('../tests/XPTO.dill', 'rb') as f:
        complete_map = dill.load(f)

except FileNotFoundError:
    complete_map = {}

try:
    with open('../tests/XPTO.dill', 'rb') as f:
        chained_decision_maps = dill.load(f)

except FileNotFoundError:
    chained_decision_maps = []

decision_maps = []


for test in range(100): #tqdm(range(10), desc='Rodada de teste '):
    for i in tqdm(range(100), desc=f'Rodada {test}'):
        model = simulate(to_repeat=False)
        decision_maps.append(model.router.decision_map)
        chained_decision_maps.append(model.router.chained_decision_map)

        if not complete_map:
            complete_map = decision_maps[0]
        else:
            for key, value in decision_maps[-1].items():
                if key in complete_map:
                    complete_map[key].extend(value)
                else:
                    complete_map[key] = value

    print(f"Estados conhecidos: {len(complete_map)}")
    ape = {len(v) for v in complete_map.values()}
    print(f"Ações por estado: {ape}")
    with open(f'../tests/decision_map_trained_with_chain_T{test}_23jun_free.dill', 'wb') as f:
        dill.dump(complete_map, f)
    del complete_map
    complete_map = {}

    with open(f'../tests/chained_decision_map_trained_T{test}_23jun_free.dill', 'wb') as f:
        dill.dump(chained_decision_maps, f)
    del chained_decision_maps
    chained_decision_maps = []
