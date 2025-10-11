import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import http.server
import socketserver
import os
import threading
import time
from datetime import datetime
import seaborn as sns


# === Ler o arquivo de log ===
with open('logs/memory/memory.log', 'r') as f:
    log_data = f.read()

# === Regex para extrair os campos ===
# pattern = re.compile(
#     r'(?P<datetime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) '
#     r'\[INFO\] mem_id=(?P<mem_id>[\d.]+) \| '
#     r'mem_size=(?P<mem_size>\d+) \| '
#     r'queues\[h\]=(?P<queues_h>[\d.]+) \| '
#     r'current_reward=(?P<current_reward>[\d.]+) \| '
#     r'cumulated_reward=(?P<cumulated_reward>[\d.]+) \| '
#     r'operated_volume=(?P<operated_volume>[\d.]+) \| '
#     r'demand=(?P<demand>[\d.]+) \| '
#     r'simulation_time\[h\]=(?P<simulation_time>[\d.]+)'
# )

# pattern = re.compile(
#     r'(?P<datetime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) '
#     r'\[INFO\] mem_id=(?P<mem_id>[A-Za-z0-9_]+) - PID \d+ \| '
#     r'mem_size=(?P<mem_size>\d+) \| '
#     r'queues\[h\]=(?P<queues_h>[\d.]+) \| '
#     r'current_reward=(?P<current_reward>[\d.]+) \| '
#     r'cumulated_reward=(?P<cumulated_reward>[\d.]+) \| '
#     r'operated_volume=(?P<operated_volume>[\d.]+) \| '
#     r'demand=(?P<demand>[\d.]+) \| '
#     r'simulation_time\[h\]=(?P<simulation_time>[\d.]+)'
# )

pattern = re.compile(
    r'(?P<datetime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) '
    r'\[INFO\] mem_id=(?P<mem_id>[A-Za-z0-9_]+ - PID \d+) \| '
    r'mem_size=(?P<mem_size>\d+) \| '
    r'queues\[h\]=(?P<queues_h>-?[\d.]+) \| '
    r'current_reward=(?P<current_reward>-?[\d.]+) \| '
    r'cumulated_reward=(?P<cumulated_reward>-?[\d.]+) \| '
    r'operated_volume=(?P<operated_volume>-?[\d.]+) \| '
    r'demand=(?P<demand>-?[\d.]+) \| '
    r'balance=(?P<balance>-?[\d.]+) \| '
    r'simulation_time\[h\]=(?P<simulation_time>-?[\d.]+)'
)


# === Extrair correspondências e criar DataFrame ===
matches = [m.groupdict() for m in pattern.finditer(log_data)]
df = pd.DataFrame(matches)

# Converter tipos
for col in df.columns:
    if col not in ['datetime', 'mem_id']:
        df[col] = df[col].astype(float)

df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S,%f")
df['velocidade'] = df['operated_volume']/df['simulation_time']
if True:
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            "Recompensa / Volume",
            "Recompensa / Fila",
            # "Tempo de Simulação"
        ],
        shared_xaxes=True
    )

    # 1️⃣ operated_volume (eixo Y principal à esquerda)
    fig.add_trace(go.Scatter(
        y=df["operated_volume"],
        x=df["queues_h"], 
        mode='markers',
        name='Operated Volume',
        marker=dict(color='blue', size=10, opacity=0.7)
    ))

    #2️⃣ queues_h (eixo Y secundário à direita)
    fig.add_trace(go.Scatter(
        y=df["velocidade"],
        x=df["queues_h"],
        mode='markers',
        name='Velocidade',
        marker=dict(color='orange', size=10, opacity=0.7),
        yaxis='y2'  # define que usa o segundo eixo Y
    ))
    

    # Ajustar layout
    fig.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=0.5, color="black")))
    fig.update_layout(template="plotly_white")

    # Mostrar interativamente
    fig.show()

resumo = df.groupby('mem_id').agg({'queues_h': 'sum', 'current_reward': 'sum', 'operated_volume': 'last', 'simulation_time': 'max', 'velocidade': 'mean'})
# resumo /= resumo.max()
# Scatter plot
fig = go.Figure()
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=[
        "Operated Volume",
        "Velocidade",
        # "Tempo de Simulação"
    ],
    shared_xaxes=True
)

# 1️⃣ operated_volume (eixo Y principal à esquerda)
fig.add_trace(go.Scatter(
    y=resumo["operated_volume"],
    x=resumo["current_reward"], 
    mode='markers',
    name='Operated Volume',
    marker=dict(color='blue', size=10, opacity=0.7)
))

#2️⃣ queues_h (eixo Y secundário à direita)
fig.add_trace(go.Scatter(
    y=resumo["velocidade"],
    x=resumo["current_reward"],
    mode='markers',
    name='Velocidade',
    marker=dict(color='orange', size=10, opacity=0.7),
    yaxis='y2'  # define que usa o segundo eixo Y
))

# Ajustar layout
fig.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=0.5, color="black")))
fig.update_layout(template="plotly_white")

# Mostrar interativamente
fig.show()

# Salvar em HTML
fig.write_html("scatter_operated_vs_reward.html", include_plotlyjs='cdn')


# 1/0
# === Gerar gráficos interativos para cada mem_id ===
if True:
    for mem_id in df['mem_id'].unique():
        reorganizado = (
            df[df['mem_id'] == mem_id]
            .groupby(['simulation_time'])
            .agg({
                'queues_h': 'last',
                'cumulated_reward': 'last',
                'operated_volume': 'last',
                'current_reward': 'last',
                'velocidade': 'last',
                'balance': 'last'
            })
            .ffill()
            .sort_index()
            .reset_index()
        )

        # === Criar figura com subplots 2x2 ===
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Queues (h)",
                "Cumulated Reward",
                "Velocidade",
                "Current Reward",
                "Volume Operado",
                "balanço de trens vazios - cheios"
            ],
            shared_xaxes=True
        )

        # === Adicionar cada gráfico scatter ===
        fig.add_trace(go.Scatter(
            x=reorganizado["simulation_time"], y=reorganizado["queues_h"],
            mode='markers', name="Queues (h)", marker=dict(color='blue', size=5)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=reorganizado["simulation_time"], y=reorganizado["cumulated_reward"],
            mode='markers', name="Cumulated Reward", marker=dict(color='orange', size=5)
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=reorganizado["simulation_time"], y=reorganizado["velocidade"],
            mode='markers', name="Operated Volume", marker=dict(color='green', size=5)
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=reorganizado["simulation_time"], y=reorganizado["current_reward"],
            mode='markers', name="Current Reward", marker=dict(color='red', size=5)
        ), row=2, col=2)

        fig.add_trace(go.Scatter(
            x=reorganizado["simulation_time"], y=reorganizado["operated_volume"],
            mode='markers', name="Volume Operado", marker=dict(color='red', size=5)
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=reorganizado["simulation_time"], y=reorganizado["balance"],
            mode='markers', name="Volume Operado", marker=dict(color='black', size=5)
        ), row=3, col=2)

        # === Layout e títulos ===
        fig.update_layout(
            height=800, width=1200,
            title_text=f"Métricas - mem_id={mem_id}",
            showlegend=False,
            template="plotly_white"
        )

        # === Salvar em HTML ===
        output_path = f"scripts/graficos/grafico_recompensa_{mem_id}.html"
        fig.write_html(output_path, include_plotlyjs='cdn')
        print(f"✅ Gráfico salvo em: {output_path}")

# --- AQUI COMEÇA A PARTE DO SERVIDOR HTTP ---

PORT = 8001
Handler = http.server.SimpleHTTPRequestHandler

def start_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Servindo em http://0.0.0.0:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()

    print("-" * 50)
    print("Servidor web iniciado!")
    print("Acesse seus gráficos no link abaixo:")
    print(f"   -> http://127.0.0.1:{PORT}/{output_path}")
    print("Pressione Ctrl+C para encerrar o servidor.")
    print("-" * 50)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServidor encerrado. Volte sempre!")