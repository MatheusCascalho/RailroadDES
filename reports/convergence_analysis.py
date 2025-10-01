import pandas as pd
import plotly.express as px
import os

df = pd.read_csv('logs/experiences/experiences_21_09_2025_22_24_56_GRAPH.log', sep=' - ')
# df = pd.read_csv('logs/logs_treinamento_07_09_2025_15_58_33.log', sep=' - ')

df.columns=['date', 'log number', 'episode', 'PID', 'volume', 'demand']
df['date'] = pd.to_datetime(df['date'])
df['volume'] = df['volume'].apply(lambda x: float(x[len('volume: '):]))
df['demand'] = df['demand'].apply(lambda x: float(x[len('demanda: '):]))

# df.set_index('date', inplace=True)
# # df['volume'].plot(kind='line', color='blue')
# fig = px.line(df, x="date", y="q_table_size", title="Convergencia do tamanho da tabela Q")
# fig.show()
# fig = px.scatter(df, x="date", y="volume", title="Convergencia do algoritmo")
fig = px.histogram(df, x='volume', color='PID')
# fig.update_traces(line_color=color)

fig.show()

fig = px.line(df, x="date", y="volume", title="Convergencia do algoritmo")
fig.show()

fig = px.scatter(df, x="date", y="volume", title="Convergencia do algoritmo")
fig.show()
 