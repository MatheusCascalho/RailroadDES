from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Union

from models.constants import Process


@dataclass
class NodeMetaData:
    node_type: str
    discretization: str

    def __post_init__(self):
        discretizations = {
            "hourly": timedelta(hours=1),
            "daily": timedelta(days=1)
        }
        if self.discretization not in discretizations:
            raise Exception(f"{self.discretization} is not a valid discretization. "
                            f"Use one of this: {discretizations.keys()}")
        self.discretization = discretizations[self.discretization]
        node_types = [
            "StockNode"
        ]
        if self.node_type not in node_types:
            raise Exception(f"{self.node_type} is not a valid node type. "
                            f"Use one of this: {node_types}")


@dataclass
class StockData:
    product: str
    capacity: float
    initial_volume: float = 0

@dataclass
class RateData:
    product: str
    rate: float

@dataclass
class ProcessorRate:
    product: str
    type: Union[str, Process]
    rate: float
    discretization: timedelta = timedelta(hours=1)

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = Process(self.type)


@dataclass
class StockNodeData:
    name: str
    stocks: list[Union[dict, StockData]]
    rates: list[Union[dict, ProcessorRate]]
    replenishment: list[Union[dict, RateData]]
    train_sizes: list[float]
    queue_capacity: int
    post_operation_time: int

    def __post_init__(self):
        self.stocks = [StockData(**s) for s in self.stocks]
        self.rates = [ProcessorRate(**s) for s in self.rates]
        self.replenishment = [RateData(**s) for s in self.replenishment]
        self.validate_products()

    def validate_products(self):
        products = [s.product for s in self.stocks]
        products_in_rates = [r.product for r in self.stocks]
        products_without_rate = [p for p in products if p not in products_in_rates]
        if products_without_rate:
            raise Exception(f"{products_without_rate} has not rates")
        products_in_replenishment = [r.product for r in self.replenishment]
        products_without_replenishment = [p for p in products if p not in products_in_replenishment]
        if self.replenishment and products_without_replenishment:
            raise Exception(f"{products_without_replenishment} has not replenishment")


@dataclass
class NodeData:
    metadata: Union[dict, NodeMetaData]
    data: Union[dict, StockNodeData]

    def __post_init__(self):
        self.metadata = NodeMetaData(**self.metadata)
        if self.metadata.node_type.__name__ == "StockNode":
            self.data = StockNodeData(**self.data)


# ##################
# {
#     "metadata": {
#         "node_type": "stocknode",
#         "discretization": "hour"
#     },
#     "data": {
#         "name": "xpto",
#         "stock": [
#             {
#                 "product": "milho",
#                 "capacity": 50e3
#             }
#         ],
#         "rates": [
#             {
#                 "product": "milho",
#                 "rate": 30e3
#             }
#         ],
#         "replenishment": [
#             {
#                 "product": "milho",
#                 "rate": 10e3
#             }
#         ]
#     }
# }
