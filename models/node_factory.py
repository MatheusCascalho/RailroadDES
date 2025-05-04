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
from models.model_queue import Queue
from models.node_constraints import ProcessConstraintSystem
from models.stock import OwnStock, StockInterface
from models.stock_replanish import SimpleStockReplanisher, ReplenishRate
from models.stock_constraints import StockToLoadTrainConstraint, StockToUnloadTrainConstraint
from models.node import StockNode
from models.maneuvering_constraints import ManeuveringConstraintFactory
from models.processors import ProcessorSystem


class AbstractNodeFactory(ABC):
    @abstractmethod
    def create_rates(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_constraints(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_queues(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_process_units(self, *args, **kwargs):
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

    def create_process_units(
            self,
            process_rates: dict[str, list[ProcessorRate]],
            queue_to_leave: Queue,
            process_constraints: list[ProcessConstraintSystem],
            process: Process
    ) -> list[ProcessorSystem]:
        process_units = [
            ProcessorSystem(
                processor_type=process,
                queue_to_leave=queue_to_leave,
                clock=self.clock,
                rates={p: rate}
            )
            for p, rates in process_rates.items()
            for rate in rates
            if rate.type == process
        ]
        for unit in process_units:
            for constraint in process_constraints:
                if constraint.process_type() == process:
                    unit.add_constraint(constraint)
        return process_units

    @staticmethod
    def create_queues(queue_capacity):
        queue_to_enter = Queue(capacity=queue_capacity, name="input")
        queue_to_leave = Queue(capacity=10_000, name="output")
        return queue_to_enter, queue_to_leave

    def create_node(self):
        stocks = self.create_stock(stock_data=self.data.stocks)
        replenishment = self.create_replenisher(replenishment=self.data.replenishment)
        constraints = self.create_constraints(
            stocks=stocks,
            train_sizes=self.data.train_sizes,
            rates=self.data.rates
        )
        maneuvering_constraint_factory = self.create_maneuver_constraint_factory(
            post_operation_time=self.data.post_operation_time
        )
        queue_to_enter, queue_to_leave = self.create_queues(self.data.queue_capacity)
        rates = self.create_rates(rates=self.data.rates)

        load_units = self.create_process_units(
            process_rates=rates,
            queue_to_leave=queue_to_leave,
            process_constraints=constraints,
            process=Process.LOAD
        )
        unload_units = self.create_process_units(
            process_rates=rates,
            queue_to_leave=queue_to_leave,
            process_constraints=constraints,
            process=Process.UNLOAD
        )

        node = StockNode(
            name=self.data.name,
            clock=self.clock,
            process_constraints=constraints,
            stocks=stocks,
            replenisher=replenishment,
            maneuvering_constraint_factory=maneuvering_constraint_factory,
            queue_to_enter=queue_to_enter,
            queue_to_leave=queue_to_leave,
            process_units=load_units+unload_units
        )
        return node


