import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

# Criar grafo com pesos
G = nx.Graph()
G.add_weighted_edges_from([
    ('A', 'B', 1.0),
    ('A', 'C', 2.5),
    ('B', 'C', 0.5),
    ('C', 'D', 3.0),
])

# Layout
pos = nx.spring_layout(G)

# Obter pesos
pesos = [G[u][v]['weight'] for u, v in G.edges()]

# Normalizar pesos para mapear para colormap
norm = colors.Normalize(vmin=min(pesos), vmax=max(pesos))
cmap = cm.get_cmap('viridis')  # outras opções: plasma, inferno, coolwarm, etc.
cores = [cmap(norm(p)) for p in pesos]

# Desenhar o grafo
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000)
nx.draw_networkx_edges(G, pos, edge_color=cores, width=3)

# Adicionar rótulo com os pesos nas arestas
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.title("Grafo com intensidade de cor nas arestas proporcional ao peso")
plt.show()
