import sys
sys.path.append(".")
import dill
import os
import numpy as np
from collections import defaultdict
import pandas as pd
from src.control.memory_based_control.reinforcement_learning.tfr_state_factory import TFRStateSpaceFactory
from src.control.memory_based_control.reinforcement_learning.action_space import ActionSpace
import networkx as nx
import matplotlib.pyplot as plt
# from pyvis.network import Network
from pprint import pprint
import itertools
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pprint import pprint
import torch
import torch.nn as nn


# Camada de projeção para dimensão 2
linear = nn.Linear(15, 2)


# with open('tests/artifacts/model_to_train_15.dill', 'rb') as f:
with open('tests/artifacts/simple_model_to_train_1_sim_v2.dill', 'rb') as f:
    model = dill.load(f)
state_space = TFRStateSpaceFactory(model)
action_space = ActionSpace(model.demands)
experiences = []
files = os.listdir()
# files = [f for f in files if f.startswith('all_experiences')]
# files = ['current/all_experiences_143591.dill']
# files = ['scripts/current/all_experiences_73673.dill']
files = ['scripts/current/all_experiences_379691.dill']
# files = ['current/all_experiences_98630.dill']
# files = [
#     "all_experiences_71113.dill",
#     "all_experiences_56652.dill",
#     "all_experiences_69867.dill",
# ]
filtered_files = []
for file in files:
    with open(file, 'rb') as f:
        exps = dill.load(f)

        if len(exps)==0 or len(exps[0].state.train_states) >3:
            continue
        experiences.extend(exps)
        filtered_files.append(file)

print(filtered_files)
experiences = [e for e in experiences if '_-origin' not in str(e.state)]
states, actions, rewards, next_states, dones = zip(*experiences)
df = pd.DataFrame(
    {"Estado": states,
     "Ação": actions,
     "Próxima Estado": next_states,
     "Reward": rewards}
)


embbed_map = {}
def to_tuple(_x, use_embedding=True):
    # try:
    # Aplicar a transformação
    x = state_space.to_array(_x)
    x = torch.from_numpy(np.array(x)).float()
    if use_embedding:
        x_embedded = linear(x)  # shape: (1, 2)
        tp = tuple([round(float(i),5) for i in tuple(x_embedded.detach().numpy())])
        embbed_map[tuple(float(t) for t in tp)] = str(_x)
        return tp
    else:
        return tuple(x.detach().numpy())
    # except Exception as e:
    #     return None

def to_scalar(a):
    try:
        # a = a if a != "ROUTING" else 'AUTOMATIC'
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

df['Estado raw'] = df['Estado']#.apply(lambda x: to_tuple(x, False))
df['Proximo Estado raw'] = df['Próxima Estado']#.apply(lambda x: to_tuple(x, False))
df['Estado'] = df['Estado'].apply(to_tuple)
df['Próxima Estado'] = df['Próxima Estado'].apply(to_tuple)

df = df.dropna(subset=['Estado'])

transitions = df.to_dict(orient='records')
filtered_transitions = []
for t in transitions:
    if len(filtered_transitions) == 0:
        filtered_transitions.append(t)
    elif t['Estado'] == filtered_transitions[-1]['Próxima Estado'] and t['Ação'] in ['AUTOMATIC', 'ROUTING']:
        filtered_transitions[-1]['Próxima Estado'] = t['Próxima Estado']
        filtered_transitions[-1]['Reward'] += t['Reward']
    else:
        filtered_transitions.append(t)

# df = pd.DataFrame(filtered_transitions)
nos = np.unique(df.dropna(subset='Próxima Estado')[['Estado', 'Próxima Estado']].values.ravel())
combinacoes = pd.DataFrame(itertools.product(nos, nos), columns=['Estado', 'Próxima Estado'])
novas = combinacoes.merge(df[['Estado', 'Próxima Estado']], on=['Estado', 'Próxima Estado'], how='left', indicator=True)
novas = novas[novas['_merge'] == 'left_only'].drop(columns=['_merge'])
novas['Ação'] = 9999999
novas['Reward'] = 9999999


# df['Locations'] = df['Estado'].apply(to_location)
# locs = {l for v in df['Locations'].values for l in v}
# df = df[df['Ação']!='AUTOMATIC']
# df = df[df['Ação']!='ROUTING']
df['Ação'] = df['Ação'].apply(to_scalar)
df['idx'] = df['Estado'].apply(str)
df_raw = df.set_index('idx')
df = df.groupby(['Estado', 'Ação', 'Próxima Estado'])['Reward'].mean().reset_index()

q_table = df.pivot(index='Estado', columns='Ação', values='Reward')
q_table = q_table[q_table[2].isna()]
q_table = q_table[q_table[3].isna()]

q_table_dict = defaultdict(lambda: defaultdict(float))

for i, row in df.iterrows():
    o = row['Estado']
    d = row['Próxima Estado']
    t = row['Ação']
    q_table_dict[o][t] = d

t = [q_table_dict[k] for k, v in q_table_dict.items() if len(v)>1]
print(set(len(v) for v in q_table_dict.values()))
df_to_graph = pd.concat([df, novas], sort=False)
df_to_graph = df_to_graph.groupby(['Estado', 'Próxima Estado']).agg({'Reward': 'mean'}).reset_index()
gain = (df_to_graph['Reward'].max()-df_to_graph['Reward'].min())
# df_to_graph['Reward']
normalized_reward = np.array(sorted((df_to_graph['Reward']).unique()))

df_to_graph['Reward'] = df_to_graph['Reward'].apply(lambda x: x if x ==9999999 else np.where(normalized_reward==x)[0][0])
# df_to_graph['Reward'] = df_to_graph['Reward']/gain

df_to_graph_2 = df_to_graph[df_to_graph['Reward']!=9999999]
simulation_map_2 = df_to_graph_2.pivot(index='Estado', columns='Próxima Estado', values='Reward')
# simulation_map_2 = simulation_map_2[list(simulation_map_2.index)]
df_to_graph['s1'] = df_to_graph['Estado'].apply(lambda x: '_'.join([str(v) for v in x]))
df_to_graph['s2'] = df_to_graph['Próxima Estado'].apply(lambda x: '_'.join([str(v) for v in x]))
simulation_map = df_to_graph.pivot(index='s1', columns='s2', values='Reward')
simulation_map = simulation_map[list(simulation_map.index)]


G = nx.from_pandas_adjacency(simulation_map.fillna(0))
# G = nx.Graph(q_table_dict)
# pos = nx.spring_layout(G)
pesos = [G[u][v]['weight'] for u, v in G.edges()]
#
# # Normalizar pesos para mapear para colormap
norm = colors.Normalize(vmin=min(df['Reward']), vmax=max(df['Reward']))
cmap = cm.get_cmap('coolwarm')  # outras opções: plasma, inferno, coolwarm, etc.
# # cores = [cmap(norm(p)) for p in pesos]
# # cores = [(0, 0, 1, 0.3) if p else 'white' for p in pesos]
# cores = ['black' if p>=0 else 'white' for p in pesos]
#
# # Desenhar o grafo
# nx.draw(G, pos, with_labels=False, node_color='lightblue')
# nx.draw_networkx_edges(G, pos, edge_color=cores, alpha=1, width=1)
#
# # Adicionar rótulo com os pesos nas arestas
# # labels = nx.get_edge_attributes(G, 'weight')
# # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#
# plt.title("Grafo com intensidade de cor nas arestas proporcional ao peso")
# plt.show()





# # Criar visualização interativa
# net = Network(notebook=False)
# net.from_nx(G)
#
# # Salvar e abrir no navegador
# net.show("grafo.html")

# print(df.head())
#
#
import plotly.graph_objects as go

pos = nx.kamada_kawai_layout(G)


# Extrair coordenadas
x_nodes = [
    float(i.split('_')[0])
    for i in G.nodes()
]
y_nodes = [float(i.split('_')[1]) for i in G.nodes()]
edge_x = []
edge_y = []

edge_traces = []
selected_colors = set()

cores_rgba = [
    (31/255, 119/255, 180/255, 1.0),  # azul
    (255/255, 127/255, 14/255, 1.0),  # laranja
    (44/255, 160/255, 44/255, 1.0),   # verde
    (214/255, 39/255, 40/255, 1.0),   # vermelho
    (148/255, 103/255, 189/255, 1.0), # roxo
    (140/255, 86/255, 75/255, 1.0),   # marrom
    (227/255, 119/255, 194/255, 1.0), # rosa
    (127/255, 127/255, 127/255, 1.0), # cinza
    (188/255, 189/255, 34/255, 1.0),  # amarelo oliva
    (23/255, 190/255, 207/255, 1.0),  # ciano

    (255/255, 0/255, 0/255, 1.0),     # vermelho puro
    (0/255, 255/255, 0/255, 1.0),     # verde puro
    (0/255, 0/255, 255/255, 1.0),     # azul puro
    (255/255, 255/255, 0/255, 1.0),   # amarelo
    (255/255, 0/255, 255/255, 1.0),   # magenta
    (0/255, 255/255, 255/255, 1.0),   # ciano
    (128/255, 0/255, 0/255, 1.0),     # vinho
    (0/255, 128/255, 0/255, 1.0),     # verde escuro
    (0/255, 0/255, 128/255, 1.0),     # azul marinho
    (255/255, 165/255, 0/255, 1.0),   # laranja claro

    (255/255, 192/255, 203/255, 1.0), # rosa claro
    (75/255, 0/255, 130/255, 1.0),    # índigo
    (255/255, 215/255, 0/255, 1.0),   # dourado
    (0/255, 100/255, 0/255, 1.0),     # floresta
    (123/255, 104/255, 238/255, 1.0), # roxo médio
    (255/255, 105/255, 180/255, 1.0), # pink forte
    (205/255, 92/255, 92/255, 1.0),   # vermelho indiano
    (240/255, 230/255, 140/255, 1.0), # cáqui
    (32/255, 178/255, 170/255, 1.0),  # verde água
    (70/255, 130/255, 180/255, 1.0),  # aço azul

    (199/255, 21/255, 133/255, 1.0),  # violeta médio
    (176/255, 224/255, 230/255, 1.0), # azul claro
    (105/255, 105/255, 105/255, 1.0), # cinza escuro
    (218/255, 165/255, 32/255, 1.0),  # dourado escuro
    (95/255, 158/255, 160/255, 1.0),  # cadet blue
    (173/255, 216/255, 230/255, 1.0)  # azul claro (light blue)

]

cores_hex = [
    "#1f77b4",  # azul
    "#ff7f0e",  # laranja
    "#2ca02c",  # verde
    "#d62728",  # vermelho
    "#9467bd",  # roxo
    "#8c564b",  # marrom
    "#e377c2",  # rosa
    "#7f7f7f",  # cinza
    "#bcbd22",  # amarelo oliva
    "#17becf",  # ciano

    "#ff0000",  # vermelho puro
    "#00ff00",  # verde puro
    "#0000ff",  # azul puro
    "#ffff00",  # amarelo
    "#ff00ff",  # magenta
    "#00ffff",  # ciano
    "#800000",  # vinho
    "#008000",  # verde escuro
    "#000080",  # azul marinho
    "#ffa500",  # laranja claro

    "#ffc0cb",  # rosa claro
    "#4b0082",  # índigo
    "#ffd700",  # dourado
    "#006400",  # floresta
    "#7b68ee",  # roxo médio
    "#ff69b4",  # pink forte
    "#cd5c5c",  # vermelho indiano
    "#f0e68c",  # cáqui
    "#20b2aa",  # verde água
    "#4682b4",  # aço azul

    "#c71585",  # violeta médio
    "#b0e0e6",  # azul claro
    "#696969",  # cinza escuro
    "#daa520",  # dourado escuro
    "#5f9ea0",  # cadet blue
    "#add8e6"  # azul claro (light blue)

]
df['chave'] = df[['Estado', 'Próxima Estado']].apply(lambda x: '_'.join([str(v) for c in x for v in c]), axis=1)
reward_map = df.set_index('chave')
edges_df = simulation_map.stack().reset_index()
for i, edge_row in edges_df.iterrows():
    s1 = edge_row['s1']
    s2 = edge_row['s2']
    a = edge_row[0]
    edge = (s1, s2)
    if a == 9999999:
        continue
    x0, y0 = [float(f) for f in s1.split('_')]
    x1, y1 = [float(f) for f in s2.split('_')]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
    # peso = int(G[edge[0]][edge[1]]['weight'])
    # peso = np.where(normalized_reward == peso)[0][0]
    try:
        k = '_'.join(edge)
        peso = reward_map.loc[k]['Reward']
        w = reward_map.loc[k]['Ação']
        if not isinstance(peso, float):
            peso = peso.iloc[0]
            w = w.iloc[0]
    except KeyError:
        k = '_'.join(edge[::-1])
        peso = reward_map.loc[k]['Reward']
        w = reward_map.loc[k]['Ação']

        if not isinstance(peso, float):
            peso = peso.iloc[0]
            w = w.iloc[0]

    # try:
    cor_rgba = cmap(norm(peso))
    # cor_rgba = cores_rgba[peso]
    cor_hex = mcolors.to_hex(cor_rgba)
    # except:
    #     pass
    # cor_hex = cores_hex[peso]
    selected_colors.add(cor_hex)

    trace = go.Scatter(
        x=edge_x[-3:],
        y=edge_y[-3:],
        mode='lines',
        line=dict(width=2,color=cor_hex),
        hoverinfo='text',
        hovertext=f'peso: {peso}',
        showlegend=False
    )
    edge_traces.append(trace)
print(len(selected_colors))
pprint(selected_colors)
# Criar figura
fig = go.Figure()

# # Arestas
# for edge in edge_traces:
#     fig.add_trace(trace)
# fig.add_trace(go.Scatter(
#     x=edge_x, y=edge_y,
#     line=dict(width=1, color='gray'),
#     hoverinfo='none',
#     mode='lines'
# ))

# Nós
estados = [embbed_map[x,y].replace('\n', '<br>') for x,y in zip(x_nodes, y_nodes)]
node_trace = go.Scatter(
    x=x_nodes, y=y_nodes,
    mode='markers',
    marker=dict(size=20, color='skyblue'),
    text=estados,
    # textposition="bottom center"
)

# Figura
fig = go.Figure(data=edge_traces + [node_trace],
                layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False),
                    margin=dict(l=0, r=0, b=0, t=40),
                    title="Grafo com cor das arestas proporcional ao peso"
                ))

fig.update_layout(showlegend=False)
fig.show()
