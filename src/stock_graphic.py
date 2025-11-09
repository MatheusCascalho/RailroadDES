from src.node import StockNode
import plotly.express as px

class StockGraphic:
    def __init__(self, nodes: list[StockNode]):
        self.nodes = nodes

    def get_history(self, node: StockNode, product: str):
        history = node.stocks[product].history().to_dataframe()
        return history

    def get_figures(self):
        figures = {}
        for node in self.nodes:
            for product in node.stocks:
                history = self.get_history(node, product)
                if history is not None:
                    # history['']
                    fig = px.line(history.reset_index(), x="instant", y="volume", title=f"Estoque de {product} no nó {node.name}")
                    figures[node.name] = fig
        # fig.show()
        return figures