from abc import abstractmethod
from src.simulation.clock import Clock
from src.domain.exceptions import StockException
from src.domain.constants import EventName, EPSILON
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from src.domain.observers import AbstractSubject, Gossiper, SubjectNotifier, to_notify
import pandas as pd


@dataclass
class StockEvent:
    event: EventName
    instant: datetime
    volume: float

@dataclass
class StockEventPromise:
    event: EventName
    promise_date: datetime
    completion_date: datetime
    volume: float

    def __post_init__(self):
        if self.completion_date <= self.promise_date:
            raise ValueError(
                f"Completion date ({self.completion_date}) must be after promise date ({self.promise_date})")

    def partial_event(self, current_date, max_volume, update=True) -> StockEvent:
        elapsed_time = current_date - self.promise_date
        complete_time = self.completion_date - self.promise_date
        if complete_time.total_seconds() == 0:
            complete_time = timedelta(hours=EPSILON) ## TODO: Melhorar essa lógica!
        reason = min(elapsed_time/complete_time, 1)
        partial_volume = self.volume * reason
        partial_volume = min(partial_volume, max_volume)
        event = StockEvent(
            event=self.event,
            instant=current_date,
            volume=partial_volume
        )
        if update and partial_volume:
            self.volume = self.volume - partial_volume
            self.promise_date = current_date
            if self.promise_date > self.completion_date:
                self.completion_date = current_date + timedelta(hours=1)
        return event

    @property
    def is_done(self):
        return self.volume < EPSILON

class StockHistory:
    def __init__(self, events):
        self.events = events

    def to_dataframe(self):
        data = [asdict(e) for e in self.events]
        if not data:
            return
        df = pd.DataFrame(data)
        df['event'] = df['event'].apply(lambda x: x.value)
        df = df.groupby(['instant', 'event']).sum().unstack(1).fillna(0)
        df.columns = df.columns.droplevel()
        if 'Dispatch Volume in Stock' in df.columns:
            df['volume'] = (df['Receive Volume in Stock'] - df['Dispatch Volume in Stock']).cumsum()
        else:
            df['volume'] = df['Receive Volume in Stock'].cumsum()

        return df

class StockInterface(AbstractSubject):
    def __init__(self):
        self.events: list[StockEvent] = []
        super().__init__()

    @to_notify()
    @abstractmethod
    def receive(self, volume: float):
        pass

    @to_notify()
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

    @property
    def product(self):
        return 0

    def get_last_receive_event(self) -> StockEvent:
        try:
            event = next(e for e in self.events[::-1] if e.event == EventName.RECEIVE_VOLUME)
            return event
        except:
            raise StockException.no_receive_event()

    @abstractmethod
    def update_promises(self):
        pass

    @abstractmethod
    def save_promise(self, promises):
        pass

    @abstractmethod
    def to_json(self):
        pass

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
        self._product = product
        self._volume = 0# initial_volume
        self.promises: list[StockEventPromise] = []
        super().__init__()
        if initial_volume > 0:
            self.receive(volume=initial_volume)

    @property
    def product(self):
        return self._product

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, v):
        self._volume = v

    @property
    def space(self):
        return self.capacity - self._volume

    @to_notify()
    def receive(self, volume: float):
        if volume == 0:
            return
        if volume < 0:
            raise ValueError("Volume must be positive")
        if self.volume + volume > self.capacity:
            StockException.stock_is_full()
        self.volume += volume
        event = StockEvent(
            event=EventName.RECEIVE_VOLUME,
            instant=self.clock.current_time,
            volume=volume
        )
        self.events.append(event)

    @to_notify()
    def dispatch(self, volume: float):
        if volume == 0:
            return
        if volume < 0:
            raise ValueError("Volume must be positive")
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
        return f"Own Stock - Volume: {round(self.volume,4)} - {round(self.volume/self.capacity, 4)*100} %"

    def save_promise(self, promises: list[StockEventPromise]):
        self.promises.extend(promises)

    def update_promises(self):
        for promise in self.promises:
            is_receive = promise.event == EventName.RECEIVE_VOLUME
            max_volume = self.space if is_receive else self.volume
            event = promise.partial_event(
                current_date=self.clock.current_time,
                max_volume=max_volume
            )
            method = self.receive if is_receive else self.dispatch
            method(event.volume)
        self.promises = [p for p in self.promises if not p.is_done]

    def to_json(self):
        return {
            "initial_volume": self.volume,
            "capacity": self.capacity,
            "product": self.product
        }


if __name__=="__main__":
    fofoqueiro_1 = Gossiper()
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

    fofoqueiro_2 = Gossiper()
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




