from models.task import Task
from dataclasses import asdict
import pandas as pd
import plotly.express as px


class OperatedVolume:
    def __init__(self, tasks: list[Task]):
        self.tasks = tasks

    def operated_volume_history(self):
        history = []
        for task in self.tasks:
            flow = asdict(task.demand.flow)
            flow['operated'] = task.demand.operated
            flow['instant'] = task.invoiced_volume_time
            history.append(flow)
        data = pd.DataFrame(history)
        return data

    def cumulated_volume_history(self):
        history = self.operated_volume_history()
        history = history.sort_values('instant').dropna().groupby('instant')['operated'].sum().cumsum().reset_index()
        return history

    def plot_operated_volume(self):
        history = self.cumulated_volume_history()
        fig = px.line(history, x="instant", y="operated", title="Volume Operado")
        # fig.show()
        return fig

    def operated_volume_by_flow(self):
        history = self.operated_volume_history()
        history = history.groupby(['origin', 'destination', 'product'])['operated'].sum().reset_index()
        return history

