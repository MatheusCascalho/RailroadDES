from abc import ABC, abstractmethod
from datetime import timedelta
from models.clock import Clock
from models.data_model import (
    StockNodeData,
    StockData,
    RateData,
    ProcessorRate
)
from models.constants import Process
from models.stock import OwnStock, StockInterface
from models.stock_replanish import SimpleStockReplanisher, ReplenishRate
from models.stock_constraints import StockToLoadTrainConstraint, StockToUnloadTrainConstraint
from models.node import StockNode
from models.maneuvering_constraints import ManeuveringConstraintFactory


class AbstractNodeFactory(ABC):
    @abstractmethod
    def create_rates(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_constraints(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_node(self, *args, **kwargs):
        pass


class NodeStockFactory(AbstractNodeFactory):
    def __init__(
            self,
            clock: Clock,
            data: StockNodeData,
    ):
        self.clock = clock
        self.data = data
        self.discretization = self.clock.discretization

    def create_rates(self, rates: list[ProcessorRate]):
        node_rates = {
            r.product: [r]
            for r in rates
        }
        return node_rates

    def create_constraints(self, stocks: list[StockInterface], train_sizes: list[float], rates: list[ProcessorRate]):
        const_type = {
            Process.LOAD: StockToLoadTrainConstraint,
            Process.UNLOAD: StockToUnloadTrainConstraint
        }
        constraints = []
        for rate in rates:
            for size in train_sizes:
                constraint = const_type[rate.type](train_size=size)
                constraints.append(constraint)
                for stock in stocks:
                    stock.add_observers([constraint])

        return constraints

    def create_stock(self, stock_data: list[StockData]):
        stocks = []
        for stock in stock_data:
            s = OwnStock(
                clock=self.clock,
                capacity=stock.capacity,
                product=stock.product,
                initial_volume=stock.initial_volume,
            )
            stocks.append(s)
        return stocks

    def create_replenisher(self, replenishment: list[RateData]):
        replenisher = SimpleStockReplanisher(
            replenish_rates=[
                ReplenishRate(
                    product=r.product,
                    rate=r.rate,
                    discretization=self.discretization
                ) for r in replenishment
            ],
            clock=self.clock
        )
        return replenisher

    def create_maneuver_constraint_factory(self, post_operation_time: int):
        factory = ManeuveringConstraintFactory(
            post_operation_time=post_operation_time,
            clock=self.clock
        )
        return factory


    def create_node(self):
        stocks = self.create_stock(stock_data=self.data.stocks)
        replenishment = self.create_replenisher(replenishment=self.data.replenishment)
        rates = self.create_rates(rates=self.data.rates)
        constraints = self.create_constraints(
            stocks=stocks,
            train_sizes=self.data.train_sizes,
            rates=self.data.rates
        )
        maneuvering_constraint_factory = self.create_maneuver_constraint_factory(
            post_operation_time=self.data.post_operation_time
        )
        node = StockNode(
            name=self.data.name,
            clock=self.clock,
            process_rates=rates,
            process_constraints=constraints,
            stocks=stocks,
            replenisher=replenishment,
            queue_capacity=self.data.queue_capacity,
            maneuvering_constraint_factory=maneuvering_constraint_factory
        )
        return node


