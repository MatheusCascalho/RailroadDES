from abc import abstractmethod
from models.clock import Clock
from models.exceptions import StockException
from models.constants import EventName
from dataclasses import dataclass
from datetime import datetime, timedelta
from models.observers import AbstractSubject, Fofoqueiro


@dataclass
class StockEvent:
    event: EventName
    instant: datetime
    volume: float


class StockHistory:
    def __init__(self, events):
        self.events: events


class StockInterface(AbstractSubject):
    def __init__(self):
        self.events: list[StockEvent] = []
        super().__init__()


    @AbstractSubject.notify_at_the_end
    @abstractmethod
    def receive(self, volume: float):
        pass

    @abstractmethod
    def dispatch(self, volume: float):
        pass

    def history(self):
        h = StockHistory(self.events)
        return h

    @property
    def volume(self):
        return 0


    @property
    def space(self):
        return 0


class OwnStock(StockInterface):
    def __init__(
            self,
            clock: Clock,
            capacity: float,
            product: str,
            initial_volume: float = 0
    ):
        self.clock = clock
        self.capacity = capacity
        self.product = product
        self._volume = initial_volume
        super().__init__()

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, v):
        self._volume = v

    @property
    def space(self):
        return self.capacity - self._volume

    @StockInterface.notify_at_the_end
    def receive(self, volume: float):
        if self.volume + volume > self.capacity:
            StockException.stock_is_full()
        self.volume += volume
        event = StockEvent(
            event=EventName.RECEIVE_VOLUME,
            instant=self.clock.current_time,
            volume=volume
        )
        self.events.append(event)

    @StockInterface.notify_at_the_end
    def dispatch(self, volume: float):
        if self.volume - volume < 0:
            StockException.stock_is_empty()
        self.volume -= volume
        event = StockEvent(
            event=EventName.DISPATCH_VOLUME,
            instant=self.clock.current_time,
            volume=volume
        )
        self.events.append(event)



    def __str__(self):
        return f"Own Stock - Volume: {self.volume} - {round(self.volume/self.capacity, 4)*100} %"


if __name__=="__main__":
    fofoqueiro_1 = Fofoqueiro()
    clk = Clock(
        start=datetime(2025,4,1),
        discretization=timedelta(hours=1)
    )
    stock1 = OwnStock(
        clock=clk,
        capacity=5e6,
        product='aço',
    )
    stock1.add_observers(fofoqueiro_1)
    print("Primeiro Recebimento:\n\n")

    stock1.receive(volume=5e3)

    fofoqueiro_2 = Fofoqueiro()
    stock2 = OwnStock(
        clock=clk,
        capacity=40,
        product='pamonha',
    )
    stock2.add_observers(fofoqueiro_2)
    print("\nSegundo Recebimento:\n")
    stock2.receive(volume=5)

    stock3 = OwnStock(
        clock=clk,
        capacity=40,
        product='milho',
    )
    stock3.add_observers(fofoqueiro_2)
    print("\nTeceiro Recebimento:\n")

    stock3.receive(volume=28)
    print("\nPrimeiro despacho:\n")

    stock3.dispatch(volume=6)




