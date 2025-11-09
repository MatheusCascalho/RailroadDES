import json

import dill
import streamlit as st
import pandas as pd
import plotly.express as px
from src.control.router import RandomRouter, RepeatedRouter, ChainedHistoryRouter
from src.domain.entities.demand import Flow
from src.statistics.stock_graphic import StockGraphic
from tests.artifacts.railroad_artifacts import create_model
from src.des_simulator import DESSimulator
from tests.artifacts.stock_node_artifacts import simple_clock
from src.statistics.gantt import Gantt
from src.statistics.operated_volume import OperatedVolume
from datetime import timedelta
from src.simulation.clock import Clock
from datetime import datetime
from src.control.solution_based_control.solution_based_router import Solution, SolutionBasedRouter
from scripts.get_best_solutions import extrair_listas_best_solution
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

base_model = 'tests/artifacts/model_to_train_15_sim_v2.dill'

caminho = "logs/vns/vns_FINAL.log"  # substitua pelo caminho do seu arquivo
resultados = extrair_listas_best_solution(caminho)
print(f"{len(resultados)} listas encontradas:")
# for i, lst in enumerate(resultados, 1):
#     print(f"{i}: {lst}")


st.set_page_config(layout="wide")

# Título da página
st.title("Simulação Ferroviária")

def simulate(to_repeat: bool, sequence):
    sim = DESSimulator(clock=Clock(start=datetime(2025,4,1), discretization=timedelta(hours=1)))
    # model = create_model(sim=sim, n_trains=3)
    with open(base_model, 'rb') as f:
        model = dill.load(f)
        for up in list(model.mesh.unload_points) + list(model.mesh.load_points):
            up.stocks['product'].capacity *= 100_00
        flow_sequence = [d.flow for o in sequence for d in model.router.demands if d.flow.origin == o]
        solution = Solution(flow_sequence=flow_sequence)
        model.router = SolutionBasedRouter(
            demands=model.router.demands,
            train_size=6e3,
            railroad_mesh=model.mesh,
            initial_solution=solution
        )
        sim.clock = model.mesh.load_points[0].clock

    if to_repeat:
        with open('decisions.json', 'r') as f:
            decisions = json.load(f)
        model.router = RepeatedRouter(model.router.demands, to_repeat=[Flow(**d) for d in decisions])
    sim.simulate(model=model, time_horizon=timedelta(days=30))
    decision_map = {s: [{'penalty': t.penalty().total_seconds()/(60*60), 'reward': t.reward()} for t in model.router.decision_map[s]] for s in model.router.decision_map}


    return model

r = st.number_input("Qtd. de repetições", step=1, min_value=50)
# st.metric(label='repeticoes', value=r)
colors = ['red', 'green', 'blue', 'orange', 'purple']
def col_gen():
    while True:
        yield 'red'
        yield 'blue'
        yield 'green'
        yield 'orange'
        yield 'purple'

color = col_gen()

simulations = []
volumes = []
# Botão de simulação
if 'repeat_button' not in st.session_state:
    st.session_state.repeat_button = False

st.session_state.simulate_button = st.button("SIMULAR")
decision_maps = []
complete_map = {}

if st.button("REPETIR"):
    st.session_state.repeat_button = not st.session_state.repeat_button
if st.session_state.simulate_button:
    for i, sequence in enumerate(resultados):
        model = simulate(to_repeat=False, sequence=sequence)
        model.router.save('decisions.json')
        decision_maps.append(model.router.decision_map)
        with open('tests/model_session.dill', 'wb') as f:
            model.router.flow_sequence = None
            dill.dump(model, f)

        gantt = Gantt().build_gantt_with_all_trains(model.trains, final_date=model.mesh.load_points[0].clock.current_time)
        # st.plotly_chart(gantt)

        # Gantt().build_gantt_by_trains(model.trains)
        op_vol = OperatedVolume(model.router.completed_tasks)
        op_vol_graph = op_vol.plot_operated_volume(color=next(color))
        simulations.append(op_vol_graph)
        opvol_table = op_vol.operated_volume_by_flow()
        volumes.append({'vol':opvol_table['operated'].sum(), 'df': opvol_table})

        # if not complete_map:
        #     complete_map = decision_maps[0]
        # else:
        #     for key, value in decision_maps[-1].items():
        #         if key in complete_map:
        #             complete_map[key].extend(value)
        #         else:
        #             complete_map[key] = value

    with open('tests/decision_map_trained.dill', 'wb') as f:
        dill.dump(complete_map, f)

    with open('tests/simulations.dill', 'wb') as f:
        dill.dump(simulations, f)

    with open('tests/volumes.dill', 'wb') as f:
        dill.dump(volumes, f)

elif st.session_state.repeat_button:
    model = simulate(to_repeat=True, sequence=[])


try:
    with open('tests/model_session.dill', 'rb') as f:
        model = dill.load(f)
    with open('tests/simulations.dill', 'rb') as f:
        simulations = dill.load(f)
    with open('tests/volumes.dill', 'rb') as f:
        volumes = dill.load(f)

    gantt = Gantt().build_gantt_with_all_trains(model.trains, final_date=model.mesh.load_points[0].clock.current_time)
    gantt_trains = Gantt().build_gantt_by_trains(model.trains, final_date=model.mesh.load_points[0].clock.current_time)

    # st.plotly_chart(gantt)

    # Gantt().build_gantt_by_trains(model.trains)
    op_vol = OperatedVolume(model.router.completed_tasks)
    op_vol_graph = op_vol.plot_operated_volume(color=next(color))
    simulations.append(op_vol_graph)
    opvol_table = op_vol.operated_volume_by_flow()
    volumes.append({'vol': opvol_table['operated'].sum(), 'df': opvol_table})

    for graphic in simulations[1:]:
        for trace in graphic.data:
            simulations[0].add_trace(trace)
    opvol_table = sorted(volumes, key=lambda x: x['vol'], reverse=True)[0]['df']
    st.metric(label="Volume Operado", value=f"{opvol_table['operated'].sum():,.0f} t".replace(",", "."))


    # Cria duas colunas
    col1, col2 = st.columns(2)

    # Mostra os gráficos em colunas separadas
    with col1:
        st.plotly_chart(gantt, use_container_width=True)

    with col2:
        st.plotly_chart(simulations[0], use_container_width=True)

    st.plotly_chart(gantt_trains, use_container_width=True)
    st.subheader("📊 Aceite Ferroviário")
    st.dataframe(opvol_table)

    st.subheader("Estoque")
    sg = StockGraphic(list(model.mesh.load_points) + list(model.mesh.unload_points))
    stocks = sg.get_figures()
    node = st.selectbox("Nó", [n.name for n in model.mesh])
    # for stock in stocks:
    st.plotly_chart(stocks[node], use_container_width=True)


except Exception as e:
    st.write("Não existe modelo executado!")

# if __name__=='__main__':
#     simulate(False, [])