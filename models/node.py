from interfaces.train_interface import TrainInterface
from models.exceptions import ProcessException
from models.node_constraints import ProcessConstraintSystem
from models.processors import ProcessorSystem
from models.data_model import ProcessorRate
from models.des_simulator import DESSimulator
from models.constants import (
    Process
)
from dataclasses import dataclass
from typing import Any
from datetime import timedelta, datetime
from models.railroad import RailSegment
from interfaces.node_interce import NodeInterface
from models.model_queue import Queue
from models.clock import Clock
from models.stock_replanish import StockReplenisherInterface
from models.stock import StockInterface
from models.maneuvering_constraints import ManeuveringConstraintFactory
from collections import defaultdict
from models.entity import Entity

@dataclass
class Neighbor:
    neighbor: NodeInterface
    transit_time: float


class Node(NodeInterface):
    def __init__(
            self,
            name: Any,
            clock: Clock,
            process_constraints: list[ProcessConstraintSystem],
            maneuvering_constraint_factory: ManeuveringConstraintFactory,
            queue_to_enter: Queue,
            queue_to_leave: Queue,
            process_units: list[ProcessorSystem],
    ):
        super().__init__(name=name,clock=clock)
        self.process_constraints = process_constraints
        self.queue_to_enter = queue_to_enter
        self.queue_to_leave = queue_to_leave
        self.process_units = process_units
        self.maneuvering_constraint_factory = maneuvering_constraint_factory
        self.liberation_constraints = defaultdict(list)

    @property
    def state(self):
        input_queue = f"Queue to enter: {self.queue_to_enter.current_size}"
        idle_process = sum(p.is_idle for p in self.process_units)
        busy = len(self.process_units) - idle_process
        load_units = f"Process Units: {idle_process} idle | {busy} busy"
        output_queue = f"Queue to leave: {self.queue_to_leave.current_size}"
        s = f"{input_queue} | {load_units} | {output_queue}"
        return s

    # ====== Events ==========
    def receive(self, train: TrainInterface):
        """
        Adiciona trem à fila de entrada e registra o log de chegada
        :param train:
        :return:
        """
        self.queue_to_enter.push(train, arrive=self.clock.current_time)
        print(f'{self.clock.current_time}:: Train {train.ID} received in node {self}!')

    def dispatch(self):
        """
        Percorre a fila de saída e remove todos os trens que não estiverem com a saída bloqueada por alguma restrição
        :param train_picker:
        :return:
        """
        for train in self.liberation_constraints:
            for constraint in self.liberation_constraints[train]:
                constraint.update()
        for train in self.queue_to_leave.now():
            none_constraint_are_blocked = all(not c.is_blocked() for c in self.liberation_constraints[train.ID])
            if none_constraint_are_blocked:
                self.queue_to_leave.pop(current_time=self.clock.current_time)
                self.liberation_constraints.pop(train.ID)
                train.leave(node=self)

                # train.leave()

    def process(self, simulator: DESSimulator):
        """
        Percorre a fila de entrada e processa todos os trens que não estiverem com processamento bloqueada
        por alguma restrição
        :return:
        """
        self.pre_processing()
        for train in self.queue_to_enter.running_queue():
            process = train.current_process_name
            slot = self._find_idle_slot(process)
            if (
                    slot is None or
                    self._is_blocked_by_constraints(process)
            ):
                self.queue_to_enter.skip_process(process)
                continue
            self._start_process(train,slot,simulator)

        self.queue_to_enter.recover()
        self.pos_processing()

    def _is_blocked_by_constraints(self, process: Process):
        return any(c.is_blocked() for c in self.process_constraints if c.process_type() == process)

    def _find_idle_slot(self, process: Process):
        processors = [p for p in self.process_units if p.type == process]
        return next((p for p in processors if p.is_idle), None)

    def _start_process(self, train, slot, simulator):
        print(f'{simulator.current_date}:: Train {self.queue_to_enter.first} starts process at node {self}!')
        train = self.queue_to_enter.pop(
            current_time=simulator.current_date
        )

        slot.put_in(train=train)

        # Update state
        time = timedelta()
        # Add next event
        simulator.add_event(
            time=time,
            callback=train.process,
            simulator=simulator,
            start=simulator.current_date,
            process_time=slot.get_process_time(),
            node=self,
            slot=slot
        )

    def pos_processing(self):
        pass

    def pre_processing(self):
        pass

    def maneuver_to_dispatch(self, simulator: DESSimulator):
        self.pre_processing()
        for slot in self.process_units:
            if slot.current_train and slot.current_train.ready_to_leave:
                print(f'{simulator.current_date}:: Train {slot.current_train.ID} entering on leaving queue!')
                train = slot.clear()
                constraint = self.maneuvering_constraint_factory.create(train_id=train.ID)
                self.liberation_constraints[train.ID].append(constraint)
                simulator.add_event(
                    time=constraint.post_operation_time,
                    callback=self.dispatch
                )

        self.pos_processing()


class StockNode(Node):
    def __init__(
            self,
            name: Any,
            clock: Clock,
            process_constraints: list[ProcessConstraintSystem],
            stocks: list[StockInterface],
            replenisher: StockReplenisherInterface,
            maneuvering_constraint_factory: ManeuveringConstraintFactory,
            queue_to_enter: Queue,
            queue_to_leave: Queue,
            process_units: list[ProcessorSystem],
    ):
        super().__init__(
                name=name,
                clock=clock,
                process_constraints=process_constraints,
                maneuvering_constraint_factory=maneuvering_constraint_factory,
                queue_to_enter=queue_to_enter,
                queue_to_leave=queue_to_leave,
                process_units=process_units,
        )
        self.replenisher = replenisher
        self.stocks: dict[str, StockInterface] = {s.product: s for s in stocks}

    def pre_processing(self):
        for stock in self.stocks.values():
            stock.update_promises()
        self.replenisher.replenish(list(self.stocks.values()))

    def pos_processing(self):
        for slot in self.process_units:
            if not slot.is_idle:
                try:
                    promise = slot.promise()
                    product = slot.current_train.product
                    self.stocks[product].save_promise([promise])
                except ProcessException:
                    continue

    def time_to_try_again(self, product: str, volume: float, process: Process):
        stock = self.stocks[product]
        current_volume = stock.volume if process == Process.LOAD else stock.space
        needed_volume = volume - current_volume
        time = self.replenisher.minimum_time_to_replenish_volume(product=product, volume=needed_volume)
        return time