import pandas as pd
import plotly.express as px
import os

df = pd.read_csv('logs_treinamento_06_09_2025_23_55_26.log', sep=' - ')

df.columns=['date', 'log_type','log number', 'episode', 'PID', 'volume', 'demand']
df['date'] = pd.to_datetime(df['date'])
df['volume'] = df['volume'].apply(lambda x: float(x[len('volume: '):]))
df['demand'] = df['demand'].apply(lambda x: float(x[len('demanda: '):]))

# df.set_index('date', inplace=True)
# df['volume'].plot(kind='line', color='blue')

# fig = px.scatter(df, x="date", y="volume", title="Convergencia do algoritmo")
fig = px.histogram(df, x='volume', color='PID')
# fig.update_traces(line_color=color)

fig.show()

fig = px.line(df, x="date", y="volume", title="Convergencia do algoritmo")
fig.show()

fig = px.scatter(df, x="date", y="volume", title="Convergencia do algoritmo")
fig.show()
 