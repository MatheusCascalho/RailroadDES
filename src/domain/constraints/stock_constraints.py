from abc import abstractmethod, ABC

from src.domain.constants import Process
from src.domain.constraints.node_constraints import ProcessConstraintSystem, BlockReason
from src.domain.observers import AbstractObserver
from src.domain.entities.stock import StockInterface
from datetime import timedelta

class StockConstraint(ProcessConstraintSystem, AbstractObserver):
    @abstractmethod
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


class StockToLoadTrainConstraint(StockConstraint):
    def __init__(self, train_size: float):
        super().__init__(train_size=train_size)

    def update(self, *args):
        stock_volume = sum(s.volume for s in self.subjects)
        if stock_volume < self.train_size:
            self.state_machine.update("block")
        else:
            self.state_machine.update("release")

    def process_type(self) -> Process:
        return Process.LOAD

    def reason(self, train_size: float, try_again: timedelta, *args, **kwargs):
        stock_volume = sum(s.volume for s in self.subjects)
        reason = BlockReason(
            constraint=self.ID,
            constraint_type=self.__class__.__name__,
            reason=f"{train_size} [train size] > {stock_volume} [stock volume]",
            time_to_try_again=try_again
        )
        return reason


class StockToUnloadTrainConstraint(StockConstraint):
    def __init__(self, train_size: float):
        super().__init__(train_size=train_size)

    def update(self, *args):
        stock_space = sum(s.space for s in self.subjects)
        if stock_space < self.train_size:
            self.state_machine.update("block")
        else:
            self.state_machine.update("release")

    def process_type(self) -> Process:
        return Process.UNLOAD

    def reason(self, train_size: float, try_again: timedelta, *args, **kwargs):
        stock_space = sum(s.space for s in self.subjects)
        reason = BlockReason(
            constraint=self.ID,
            constraint_type=self.__class__.__name__,
            reason=f"{train_size} [train size] > {stock_space} [Space]",
            time_to_try_again=try_again
        )
        return reason