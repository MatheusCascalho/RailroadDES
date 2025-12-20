from datetime import timedelta
from src.domain.entities.node import Node
from src.domain.entities.train import Train
from src.events.event import EventCommand
from src.domain.exceptions import NotCompletedEvent
from src.domain.constants import Process, EventName
from src.standards.observers import to_notify


class StartProcess(EventCommand):
    def __init__(
            self,
            train: Train,
            node: Node,
            process: Process
        ) -> None:
        super().__init__()
        self.node = node
        self.train = train
        self.process = process
        
    @to_notify()
    def execute(self):
        try:
            slot = self.node.slot_assignment(train=self.train)
        except:
            raise NotCompletedEvent()
        self.node.move_to_slot(train=self.train,slot=slot)
        self.train.start_process()
        self.recorder.record(train=self.train, event_name=EventName.START_PROCESS, process=self.process)


class FinishProcess(EventCommand):
    def __init__(
        self,
        train: Train,
        node: Node,
        process: Process,
        process_time: timedelta
    ) -> None:
        super().__init__(pre_time=process_time)
        self.node = node
        self.train = train
        self.process = process


    @to_notify()
    def execute(self):
        self.train.finish_process(process=Process.LOAD)
        self.recorder.record(train=self.train, event_name=EventName.FINISH_PROCESS, process=self.process)
        self.node.maneuver_to_dispatch(train=self.train)
        
        
