from abc import ABC
from typing import Union

from interfaces.train_interface import TrainInterface
from models import model_queue as mq
from models.clock import Clock
from models.constants import Process
from models.discrete_event_system import DiscreteEventSystem
from models.state_machine import State, Transition, StateMachine, MultiCriteriaTransition
from models.states import ProcessorState, NodeProcesState
from models.stockinterface import StockInterface
from models.observers import AbstractObserver


class ProcessorSystem(DiscreteEventSystem):
    def __init__(
            self,
            processor_type: Process,
            queue_to_leave: mq.Queue,
            clock: Clock
    ):
        self.type = processor_type
        self.current_train: Union[None, TrainInterface] = None
        self.queue_to_leave = queue_to_leave
        self.clock = clock
        super().__init__()


    def build_state_machine(self):
        idle = State(name=ProcessorState.IDLE, is_marked=True)
        busy = State(name=ProcessorState.BUSY, is_marked=False)
        put_in = Transition(name="Put in", origin=idle, destination=busy)
        free_up = Transition(name="Free Up", origin=busy, destination=idle, action=self.clear)
        sm = StateMachine(transitions=[put_in, free_up])
        return sm

    def put_in(self, train: TrainInterface):
        if self.state_machine.current_state.name == ProcessorState.IDLE:
            self.current_train = train
            self.state_machine.update()
        else:
            raise Exception("Processor is Busy")

    def free_up(self):

        if self.state_machine.current_state.name == ProcessorState.BUSY:
            self.current_train = None
            self.state_machine.update()
        else:
            raise Exception("Processor is Idle")

    def clear(self):
        train = self.current_train
        self.queue_to_leave.push(
            train,
            arrive=self.clock.current_time
        )
        return train

    def is_ready_to_clear(self):
        if self.current_train:
            return self.current_train.process_end <= self.clock.current_time
        return False

    @property
    def is_idle(self):
        return self.state_machine.current_state.name == ProcessorState.IDLE


class ProcessConstraintSystem(DiscreteEventSystem):
    def __init__(
            self,
    ):
        super().__init__()


    def build_state_machine(self) -> StateMachine:
        ready = State(name=NodeProcesState.READY, is_marked=True)
        busy = State(name=NodeProcesState.BUSY, is_marked=False)
        blocked = State(name=NodeProcesState.BLOCKED, is_marked=False)

        start = MultiCriteriaTransition(
            name="start",
            origin=ready,
            destination=busy
        )
        finish = MultiCriteriaTransition(
            name="finish",
            origin=busy,
            destination=ready
        )
        block = MultiCriteriaTransition(
            name="block",
            origin=ready,
            destination=blocked
        )
        release = MultiCriteriaTransition(
            name="release",
            origin=blocked,
            destination=ready
        )
        sm = StateMachine(transitions=[
            start,finish,
            block,release
        ])
        return sm

    def is_blocked(self):
        return self.state_machine.current_state.name == NodeProcesState.BLOCKED


class StockConstraint(ProcessConstraintSystem, AbstractObserver):
    def __init__(self, train_size: float):
        self.train_size = train_size
        ProcessConstraintSystem.__init__(self)
        AbstractObserver.__init__(self)

    def append_subject(self, sub):
        if not isinstance(sub, StockInterface):
            raise Exception("StockConstraint only look for Stock Objects")
        self.subjects.append(sub)
        self.update()



class StockLoadConstraint(StockConstraint):
    def __init__(self, train_size: float):
        super().__init__(train_size=train_size)

    def update(self, *args):
        stock_volume = sum(s.volume for s in self.subjects)
        if stock_volume < self.train_size:
            self.state_machine.update("block")
        else:
            self.state_machine.update("release")


class StockUnloadConstraint(StockConstraint):
    def __init__(self, train_size: float):
        super().__init__(train_size=train_size)

    def update(self, *args):
        stock_space = sum(s.space for s in self.subjects)
        if stock_space < self.train_size:
            self.state_machine.update("block")
        else:
            self.state_machine.update("release")

if __name__=="__main__":
    from models.stockinterface import OwnStock
    from models.clock import Clock
    from datetime import datetime, timedelta

    clk = Clock(
        start=datetime(2025, 4, 1),
        discretization=timedelta(hours=1)
    )
    stock1 = OwnStock(
        clock=clk,
        capacity=50,
        product='aço',
    )
    restriction_1 = StockLoadConstraint(
        train_size=10
    )
    restriction_2 = StockLoadConstraint(
        train_size=5
    )
    stock1.add_observers([restriction_1, restriction_2])

    assert restriction_1.is_blocked()
    assert restriction_2.is_blocked()

    stock1.receive(
        volume=15
    )

    assert not restriction_1.is_blocked()
    assert not restriction_2.is_blocked()

    stock1.dispatch(
        volume=10
    )

    assert restriction_1.is_blocked()
    assert not restriction_2.is_blocked()

