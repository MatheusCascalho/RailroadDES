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

# caminho para o log
log_file = "logs/learning/Learner_0.log"

# regex para extrair métricas e o timestamp
pattern = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[INFO\] \[step=(\d+)\] ε=([\d.]+) \| loss=([\d.]+) \| reward_avg=([\d.\-]+) \| Q_avg=([\d.\-]+)"
)

data = []
with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            timestamp_str, step, epsilon, loss, reward, q_avg = match.groups()
            data.append({
                "timestamp": datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f'),
                "step": int(step),
                "epsilon": float(epsilon),
                "loss": float(loss),
                "reward_avg": float(reward),
                "Q_avg": float(q_avg),
            })

# cria DataFrame
df = pd.DataFrame(data)

# Define a coluna 'timestamp' como o índice do DataFrame
df = df.set_index('timestamp')

# Cria os gráficos separadamente, usando o índice (timestamp) no eixo X
fig1 = px.line(df, x=df.index, y="loss", title="Convergência do Loss", markers=True)
fig2 = px.line(df, x=df.index, y="reward_avg", title="Recompensa Média por Step", markers=True)
fig3 = px.line(df, x=df.index, y="Q_avg", title="Valor Médio de Q", markers=True)
fig4 = px.line(df, x=df.index, y="epsilon", title="Decaimento de Epsilon", markers=True)

# Converte os gráficos para HTML
html_content = f"""
<html>
<head>
    <title>Análise de Treinamento - DQN</title>
    <meta charset="UTF-8">
</head>
<body>
    <h1>Análise de Métricas do Treinamento DQN</h1>
    {fig1.to_html(full_html=False, include_plotlyjs='cdn')}
    <hr>
    {fig2.to_html(full_html=False, include_plotlyjs='cdn')}
    <hr>
    {fig3.to_html(full_html=False, include_plotlyjs='cdn')}
    <hr>
    {fig4.to_html(full_html=False, include_plotlyjs='cdn')}
</body>
</html>
"""

# Salva o conteúdo em um arquivo HTML
file_name = "analise_treinamento.html"
with open(file_name, "w", encoding='utf-8') as f:
    f.write(html_content)

# --- AQUI COMEÇA A PARTE DO SERVIDOR HTTP ---

PORT = 8000
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