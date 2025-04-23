from abc import abstractmethod

from models.node_constraints import ProcessConstraintSystem
from models.observers import AbstractObserver
from models.stock import StockInterface


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


class StockToUnloadTrainConstraint(StockConstraint):
    def __init__(self, train_size: float):
        super().__init__(train_size=train_size)

    def update(self, *args):
        stock_space = sum(s.space for s in self.subjects)
        if stock_space < self.train_size:
            self.state_machine.update("block")
        else:
            self.state_machine.update("release")
