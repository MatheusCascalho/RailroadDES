from datetime import timedelta
from src.domain.entities.node import Node
from src.domain.entities.train import Train
from src.events.event import EventCommand, EventFactory
from src.interfaces.des_simulator_interface import DESSimulatorInterface
from src.domain.exceptions import NotCompletedEvent, TrainExceptions, FinishedTravelException
from logging import debug
from src.domain.entities.time_table import TimeEvent
from src.domain.constants import (
    EPSILON,
    TrainActions,
    Process, EventName
)
from src.interfaces.node_interce import NodeInterface
from src.interfaces.train_interface import TrainInterface
from src.standards.observers import to_notify


class StartLoad(EventCommand):
    def __init__(
            self,
            train: Train,
            node: Node,
        ) -> None:
        super().__init__()
        self.node = node
        self.train = train
        
    @to_notify()
    def execute(self):
        try:
            slot = self.node.slot_assignment(train=self.train)
        except:
            raise NotCompletedEvent()
        self.node.move_to_slot(train=self.train,slot=slot)
        self.train.start_process()
        self.recorder.record(train=self.train, event_name=EventName.START_PROCESS, process=Process.LOAD)
        
