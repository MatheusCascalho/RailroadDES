import dill
import streamlit as st
import pandas as pd
import plotly.express as px
from tests.artifacts.railroad_artifacts import create_model
from models.des_simulator import DESSimulator
from tests.artifacts.stock_node_artifacts import simple_clock
from models.gantt import Gantt
from models.operated_volume import OperatedVolume
from datetime import timedelta
from models.clock import Clock
from datetime import datetime

st.set_page_config(layout="wide")

# Título da página
st.title("Simulação Ferroviária")

def simulate():
    sim = DESSimulator(clock=Clock(start=datetime(2025,4,1), discretization=timedelta(hours=1)))
    # model = create_model(sim=sim, n_trains=3)
    with open('tests/artifacts/model.dill', 'rb') as f:
        model = dill.load(f)
        sim.clock = model.mesh.load_points[0].clock
    sim.simulate(model=model, time_horizon=timedelta(days=20))
    decision_map = {s: [{'penalty': t.penalty().total_seconds()/(60*60), 'reward': t.reward()} for t in model.router.decision_map[s]] for s in model.router.decision_map}

    return model

r = st.number_input("Qtd. de repetições", step=1, min_value=1)
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
# Botão de simulação
if st.button("SIMULAR"):
    simulations = []
    volumes = []
    for i in range(r):
        model = simulate()

        gantt = Gantt().build_gantt_with_all_trains(model.trains)
        # st.plotly_chart(gantt)

        # Gantt().build_gantt_by_trains(model.trains)
        op_vol = OperatedVolume(model.router.completed_tasks)
        op_vol_graph = op_vol.plot_operated_volume(color=next(color))
        simulations.append(op_vol_graph)
        opvol_table = op_vol.operated_volume_by_flow()
        volumes.append({'vol':opvol_table['operated'].sum(), 'df': opvol_table})


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

    st.subheader("📊 Aceite Ferroviário")
    st.dataframe(opvol_table)
