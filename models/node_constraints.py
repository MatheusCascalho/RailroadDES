from models.discrete_event_system import DiscreteEventSystem
from models.state_machine import State, StateMachine, MultiCriteriaTransition
from models.states import NodeProcesState
from models.stock import StockInterface
from models.observers import AbstractObserver


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
    def update(self, *args):
        pass

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
    from models.stock import OwnStock
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

