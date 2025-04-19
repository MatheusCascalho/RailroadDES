from typing import Union

from networkx.algorithms.structuralholes import constraint

from interfaces.train_interface import TrainInterface
from models import model_queue as mq
from models.clock import Clock
from models.constants import Process
from models.discrete_event_system import DiscreteEventSystem
from models.state_machine import State, Transition, StateMachine
from models.states import ProcessorState
from models.node_constraints import (
    ProcessConstraintSystem,
    StockConstraint,
    StockLoadConstraint,
    StockUnloadConstraint
)
from models.stock import StockInterface


class ProcessorSystem(DiscreteEventSystem):
    def __init__(
            self,
            processor_type: Process,
            queue_to_leave: mq.Queue,
            clock: Clock,
            constraints: list[ProcessConstraintSystem] = []
    ):
        self.type = processor_type
        self.current_train: Union[None, TrainInterface] = None
        self.queue_to_leave = queue_to_leave
        self.clock = clock
        self.constraints = constraints
        super().__init__()


    def build_state_machine(self):
        idle = State(name=ProcessorState.IDLE, is_marked=True)
        busy = State(name=ProcessorState.BUSY, is_marked=False)
        put_in = Transition(name="Put in", origin=idle, destination=busy)
        free_up = Transition(name="Free Up", origin=busy, destination=idle, action=self.clear)
        sm = StateMachine(transitions=[put_in, free_up])
        return sm

    def active_constraints(self, train: TrainInterface) -> list[ProcessConstraintSystem]:
        active = []
        for c in self.constraints:
            if isinstance(c, StockConstraint) and c.train_size <= train.capacity:
                active.append(c)
        return active


    def put_in(self, train: TrainInterface):
        if all(not c.is_blocked() for c in self.active_constraints(train)):
            if self.state_machine.current_state.name == ProcessorState.IDLE:
                self.current_train = train
                self.state_machine.update()
            else:
                raise Exception("Processor is Busy")
        else:
            raise Exception("Processor is Blocked")

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


class ProcessorSystemBuilder:
    @staticmethod
    def build_stock_constraints(
            process_type: Process,
            stocks: list[StockInterface],
            train_sizes
    ):
        constraint_type = {
            Process.LOAD: StockLoadConstraint,
            Process.UNLOAD: StockUnloadConstraint
        }
        stock_constraints = [
            constraint_type[process_type](train_size=train_size)
            for train_size in train_sizes
        ]
        for stock in stocks:
            stock.add_observers(stock_constraints)
        return stock_constraints
