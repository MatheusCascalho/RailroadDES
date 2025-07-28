import dill
import os
import numpy as np
from collections import defaultdict
import pandas as pd
from models.tfr_state_factory import TFRStateSpaceFactory
from models.DQNRouter import ActionSpace
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network


with open('../tests/artifacts/model_to_train_15.dill', 'rb') as f:
    model = dill.load(f)
state_space = TFRStateSpaceFactory(model)
action_space = ActionSpace(model.demands)
experiences = []
files = os.listdir()
files = [f for f in files if f.startswith('all_experiences')]
# files = [
#     "all_experiences_71113.dill",
#     "all_experiences_56652.dill",
#     "all_experiences_69867.dill",
# ]
for file in files:
    with open(file, 'rb') as f:
        experiences.extend(dill.load(f))


states, actions, rewards, next_states, dones = zip(*experiences)
df = pd.DataFrame(
    {"Estado": states,
     "Ação": actions,
     "Próxima Estado": next_states,
     "Reward": rewards}
)
def to_tuple(x):
    try:
        return tuple(state_space.to_array(x))
    except:
        return None

def to_scalar(a):
    try:
        return action_space.to_scalar(a)
    except:
        return None

def to_location(x):
    trains = 15
    local_cardinality = 5
    positions = trains * local_cardinality
    positions = x[:positions]
    locs = set()
    for t in range(trains):
        locs.add(tuple(positions[:local_cardinality]))
        positions = positions[local_cardinality:]
    return locs

df['Estado'] = df['Estado'].apply(to_tuple)
df = df.dropna(subset=['Estado'])

df['Locations'] = df['Estado'].apply(to_location)
locs = {l for v in df['Locations'].values for l in v}
df['Próxima Estado'] = df['Próxima Estado'].apply(to_tuple)
df = df[df['Ação']!='AUTOMATIC']
df['Ação'] = df['Ação'].apply(to_scalar)
df = df.groupby(['Estado', 'Ação', 'Próxima Estado']).mean().reset_index()
q_table = df.pivot(index='Estado', columns='Ação', values='Reward')
simulation_map = df.pivot(index='Estado', columns='Próxima Estado', values='Ação')
# converted_states = []
# for s in states:
#     try:
#         s = state_space.to_array(s)
#         converted_states.append(s)
#     except:
#         continue
#
# converted_states = np.matrix(converted_states)

# q_table = {
#     tuple(cs): defaultdict(lambda: defaultdict(lambda: 0))
#     for cs in converted_states
# }
#
# for experience in experiences:

G = nx.from_numpy_array(simulation_map.values)

# nx.draw(G, with_labels=True)
# plt.show()


# Criar visualização interativa
net = Network(notebook=False)
net.from_nx(G)

# Salvar e abrir no navegador
net.show("grafo.html")

print(df.head())


import plotly.graph_objects as go

pos = nx.spring_layout(G)


# Extrair coordenadas
x_nodes = [pos[i][0] for i in G.nodes()]
y_nodes = [pos[i][1] for i in G.nodes()]
edge_x = []
edge_y = []

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

# Criar figura
fig = go.Figure()

# Arestas
fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='gray'),
    hoverinfo='none',
    mode='lines'
))

# Nós
fig.add_trace(go.Scatter(
    x=x_nodes, y=y_nodes,
    mode='markers+text',
    marker=dict(size=20, color='skyblue'),
    text=[str(i) for i in G.nodes()],
    textposition="bottom center"
))

fig.update_layout(showlegend=False)
fig.show()
