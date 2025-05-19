from models.clock import Clock
from models.stock import StockInterface
from abc import abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from models.exceptions import StockException


class StockReplenisherInterface:
    @abstractmethod
    def replenish(self, stock: list[StockInterface]):
        pass

    @abstractmethod
    def minimum_time_to_replenish_volume(self, *args, **kwargs):
        pass

    @abstractmethod
    def to_json(self):
        pass


@dataclass
class ReplenishRate:
    product: str
    rate: float
    discretization: timedelta = timedelta(hours=1)

class SimpleStockReplanisher(StockReplenisherInterface):
    def __init__(
            self,
            replenish_rates: list[ReplenishRate],
            clock: Clock
    ):
        self.replenish_rates: dict[str, ReplenishRate] = {
            r.product: r
            for r in replenish_rates
        }
        self.clock = clock

    def replenish(
            self,
            stocks: list[StockInterface]
    ):
        for stock in stocks:
            if stock.product not in self.replenish_rates:
                raise Exception(f'Taxa do produto {stock.product} não foi cadastrada!!')
            rate = self.replenish_rates[stock.product]
            try:
                last_event = stock.get_last_receive_event()
                elapsed_time = self.clock.current_time - last_event.instant
            except StockException:
                elapsed_time = self.clock.elapsed_time()
            steps = elapsed_time/rate.discretization
            if steps:
                volume = rate.rate * steps
                volume = min(volume, stock.space)
                stock.receive(
                    volume=volume
                )

    def minimum_time_to_replenish_volume(self, product: str, volume: float) -> timedelta:
        rate = self.replenish_rates[product]
        time = volume / rate.rate
        return timedelta(hours=time)

    def to_json(self):
        return dict(
            replenish_rates=[r for r in self.replenish_rates.values()]
        )

