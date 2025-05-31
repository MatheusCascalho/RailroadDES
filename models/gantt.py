import pandas as pd
from models.time_table import TimeRegister, TimeTable
import plotly.express as px

class Gantt:
    def __init__(self):
        pass

    @staticmethod
    def to_dataframe(time_table: TimeTable):
        data = []
        for register in time_table.registers:
            rows = [
                {
                    'Process': 'Fila de Entrada',
                    'Start': register.arrive.instant,
                    'Finish': register.start_process.instant,
                    'Location': register.register_location
                },
                {
                    'Process': 'Processo',
                    'Start': register.start_process.instant,
                    'Finish': register.finish_process.instant,
                    'Location': register.register_location
                },
                {
                    'Process': 'Fila de Saída',
                    'Start': register.finish_process.instant,
                    'Finish': register.departure.instant,
                    'Location': register.register_location
                },

            ]
            data.extend(rows)
        return pd.DataFrame(data)

    def build_gantt(self, time_table: TimeTable):
        df = self.to_dataframe(time_table)
        # Criando o gráfico de Gantt
        fig = px.timeline(df, x_start="Start", x_end="Finish", y="Location", color="Process", title="Diagrama de Gantt")
        fig.update_yaxes(categoryorder="total ascending")  # Organiza as tarefas pela ordem total
        fig.show()

    def build_gantt_with_all_trains(self, trains):
        dfs = []
        for train in trains:
            time_table = train.time_table
            df = self.to_dataframe(time_table)
            df['Location'] += f" Trem {train.ID}"
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True).sort_values('Location')
        # Criando o gráfico de Gantt
        fig = px.timeline(df, x_start="Start", x_end="Finish", y="Location", color="Process", title="Diagrama de Gantt")
        # fig.update_yaxes(categoryorder="total ascending")  # Organiza as tarefas pela ordem total
        # Definindo os ticks do eixo Y
        loc = {}
        for location in df['Location'].unique():
            loc[location[:-len("Trem train_xx")].strip()] = location

        fig.update_layout(
            yaxis=dict(
                tickvals=list(loc.values()),  # Definindo as tarefas específicas para aparecer no eixo Y
                ticktext=list(loc.keys())  # Alterando os rótulos dessas tarefas
            )
        )

        # fig.show()
        return fig

    def build_gantt_by_trains(self, trains):
        dfs = []
        for train in trains:
            time_table = train.time_table
            df = self.to_dataframe(time_table)
            # df['Location'] += f" Trem {train.ID}"
            df['Train'] = train.ID
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True).sort_values('Location')
        # Criando o gráfico de Gantt
        fig = px.timeline(df, x_start="Start", x_end="Finish", y="Train", color="Location", title="Diagrama de Gantt Para trens")
        # fig.update_yaxes(categoryorder="total ascending")  # Organiza as tarefas pela ordem total
        # Definindo os ticks do eixo Y
        loc = {}
        for location in df['Location'].unique():
            loc[location[:-len("Trem train_xx")]] = location

        # fig.update_layout(
        #     yaxis=dict(
        #         tickvals=list(loc.values()),  # Definindo as tarefas específicas para aparecer no eixo Y
        #         ticktext=list(loc.keys())  # Alterando os rótulos dessas tarefas
        #     )
        # )

        return fig


if __name__ == '__main__':
    # import plotly.express as px
    # import pandas as pd
    #
    # # Exemplo de dados para o diagrama de Gantt
    # data = {
    #     'Task': ['Task 1', 'Task 1', 'Task 3', 'Task 4'],
    #     'Start': ['2025-05-01', '2025-05-05', '2025-05-10', '2025-05-12'],
    #     'Finish': ['2025-05-05', '2025-05-10', '2025-05-12', '2025-05-15'],
    #     'Resource': ['Team A', 'Team B', 'Team A', 'Team C']
    # }
    #
    # # Criando o dataframe
    # df = pd.DataFrame(data)
    #
    # # Convertendo as datas para o formato correto
    # df['Start'] = pd.to_datetime(df['Start'])
    # df['Finish'] = pd.to_datetime(df['Finish'])
    #
    # # Criando o gráfico de Gantt
    # fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Resource", title="Diagrama de Gantt")
    # fig.update_yaxes(categoryorder="total ascending")  # Organiza as tarefas pela ordem total
    # fig.show()

    import plotly.express as px
    import pandas as pd

    # Criando os dados
    data = {
        "Task": ["C444", "Reg", "Reg", "C845", "C98"],
        "Start": ["2025-05-01 06:00", "2025-05-01 12:00", "2025-05-01 18:00", "2025-05-02 06:00", "2025-05-02 12:00"],
        "Finish": ["2025-05-01 12:00", "2025-05-01 18:00", "2025-05-02 00:00", "2025-05-02 12:00", "2025-05-03 00:00"],
        "Resource": ["EGN-Terminal", "Reg", "Reg", "C845", "C98"]
    }

    df = pd.DataFrame(data)

    # Convertendo para o formato de data
    df['Start'] = pd.to_datetime(df['Start'])
    df['Finish'] = pd.to_datetime(df['Finish'])

    # Gerando o gráfico de Gantt
    fig = px.timeline(df,
                      x_start="Start",
                      x_end="Finish",
                      y="Task",
                      color="Resource",
                      title="Diagrama de Gantt - Exemplo")

    # Ajustando o layout para melhorar a visualização
    fig.update_layout(xaxis_title="Data", yaxis_title="Tarefas", showlegend=True)

    # Exibindo o gráfico
    fig.show()





