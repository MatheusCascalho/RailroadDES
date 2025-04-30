from typing import Union

from datetime import timedelta
from interfaces.train_interface import TrainInterface
from models import model_queue as mq
from models.clock import Clock
from models.constants import Process, EventName
from models.data_model import ProcessorRate
from models.discrete_event_system import DiscreteEventSystem
from models.exceptions import ProcessException
from models.state_machine import State, Transition, StateMachine
from models.states import ProcessorState
from models.node_constraints import (
    ProcessConstraintSystem
)
from models.stock_constraints import StockConstraint, StockToLoadTrainConstraint, StockToUnloadTrainConstraint
from models.stock import StockInterface, StockEventPromise


class ProcessorSystem(DiscreteEventSystem):
    def __init__(
            self,
            processor_type: Process,
            queue_to_leave: mq.Queue,
            clock: Clock,
            rates: dict[str, ProcessorRate],
            constraints: list[ProcessConstraintSystem] = []
    ):
        self.type = processor_type
        self.current_train: Union[None, TrainInterface] = None
        self.queue_to_leave = queue_to_leave
        self.clock = clock
        self.constraints = constraints
        self.rates = rates
        self.promised = False
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
                train.add_to_slot()
                self.state_machine.update()
            else:
                raise ProcessException.process_is_busy()
        else:
            raise ProcessException.process_is_blocked()

    def free_up(self):
        if self.state_machine.current_state.name == ProcessorState.BUSY:
            self.current_train = None
            self.promised = False
            self.state_machine.update()
        else:
            raise Exception("Processor is Idle")

    def clear(self):
        train = self.current_train
        train.removed_from_slot()
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

    def promise(self) -> StockEventPromise:
        if not self.is_idle and not self.promised:
            product = self.current_train.product
            time_to_finish = self.get_process_time()
            completion_date = self.clock.current_time + time_to_finish
            event = StockEventPromise(
                event=EventName.DISPATCH_VOLUME if self.type == Process.LOAD else EventName.RECEIVE_VOLUME,
                promise_date=self.clock.current_time,
                completion_date=completion_date,
                volume=self.current_train.capacity
            )
            self.promised = True
            return event
        raise ProcessException.no_promise_to_do()

    def get_process_time(self) -> timedelta():
        if not self.is_idle:
            volume = self.current_train.capacity
            product = self.current_train.product
            rate = self.rates[product].rate
            steps = volume / rate
            process_time = steps * self.rates[product].discretization
            return process_time
        return timedelta()

    def add_constraint(self, constraint: ProcessConstraintSystem):
        self.constraints.append(constraint)

        # When process starts, constraint is updated to busy state
        self.state_machine.states[ProcessorState.BUSY].add_observers(
            constraint.state_machine.transitions["start"]
        )
        # When process finish, constraint is updated to ready state
        self.state_machine.states[ProcessorState.IDLE].add_observers(
            constraint.state_machine.transitions["finish"]
        )



class ProcessorSystemBuilder:
    @staticmethod
    def build_stock_constraints(
            process_type: Process,
            stocks: list[StockInterface],
            train_sizes
    ):
        constraint_type = {
            Process.LOAD: StockToLoadTrainConstraint,
            Process.UNLOAD: StockToUnloadTrainConstraint
        }
        stock_constraints = [
            constraint_type[process_type](train_size=train_size)
            for train_size in train_sizes
        ]
        for stock in stocks:
            stock.add_observers(stock_constraints)
        return stock_constraints
