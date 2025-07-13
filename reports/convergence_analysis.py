import pandas as pd
import plotly.express as px

df = pd.read_csv('treinamento_42x6.csv', sep=';')

df.columns=['date', 'log_type', 'episode', 'volume', 'demand']
df['date'] = pd.to_datetime(df['date'])
df['volume'] = df['volume'].apply(lambda x: float(x[len('volume: '):]))
df['demand'] = df['demand'].apply(lambda x: float(x[len('demanda: '):]))

# df.set_index('date', inplace=True)
# df['volume'].plot(kind='line', color='blue')

fig = px.line(df, x="date", y="volume", title="Convergencia do algoritmo")
# fig.update_traces(line_color=color)

fig.show()