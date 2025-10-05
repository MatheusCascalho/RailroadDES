import pandas as pd
import plotly.express as px
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


df = pd.read_csv('logs/vns/vns_05out_17h.log', sep='|').reset_index()
df.columns=['time', 'neighborhood', 'neitgborhood_type', 'fitness', 'is_local_optimal', 'start', 'end', 'simulation_time', 'c']
df['fitness'] = df['fitness'].apply(lambda x:None if 'fitness_logistic_time' not in x else float(x[len('fitness_logistic_time=')+1:]))
df['is_local_optimal'] = df['is_local_optimal'].apply(lambda x:False if 'False' in x else True)
df['simulation_time'] = df['simulation_time'].apply(lambda x:float(x[len('simulation_time=')+1:]))
df['time'] = pd.to_datetime(df['time'].apply(lambda x:x[:len('2025-10-05 12:00:23,310')]))

# Define a coluna 'timestamp' como o índice do DataFrame
df = df.set_index('time')

# Cria os gráficos separadamente, usando o índice (timestamp) no eixo X
fig1 = px.line(df, x=df.index, y="fitness", title="Busca de melehor tempo", markers=True)
fig2 = px.line(df[df['is_local_optimal']], x=df[df['is_local_optimal']].index, y="fitness", title="Busca de melehor tempo", markers=True)
fig3 = px.histogram(df, x="simulation_time", title="Tempo de simulação")
# fig3 = px.line(df, x=df.index, y="Q_avg", title="Valor Médio de Q", markers=True)
# fig4 = px.line(df, x=df.index, y="epsilon", title="Decaimento de Epsilon", markers=True)

# Converte os gráficos para HTML
html_content = f"""
<html>
<head>
    <title>Análise de Convergência - VNS</title>
    <meta charset="UTF-8">
</head>
<body>
    <h1>Análise de Métricas do Treinamento DQN</h1>
    {fig1.to_html(full_html=False, include_plotlyjs='cdn')}
    <hr>
    {fig2.to_html(full_html=False, include_plotlyjs='cdn')}
    <hr>
    {fig3.to_html(full_html=False, include_plotlyjs='cdn')}
</body>
</html>
"""

# Salva o conteúdo em um arquivo HTML
file_name = "analise_vns.html"
with open(file_name, "w", encoding='utf-8') as f:
    f.write(html_content)


# --- AQUI COMEÇA A PARTE DO SERVIDOR HTTP ---

PORT = 8002
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
    print(f"   -> http://127.0.0.1:{PORT}/{file_name}")
    print("Pressione Ctrl+C para encerrar o servidor.")
    print("-" * 50)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServidor encerrado. Volte sempre!")