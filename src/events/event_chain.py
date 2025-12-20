from abc import abstractmethod, ABC
from datetime import timedelta

from src.domain.entities.node import Node
from src.domain.entities.train import Train
from src.events.event import EventCommand
from src.events.event_queue import EventQueue
from src.events.load_events import FinishLoad, StartLoad
from src.interfaces.train_interface import TrainInterface
from src.simulation.event_calendar import EventCalendar
from src.standards.observers import AbstractObserver, AbstractSubject, to_notify

class EventChain(AbstractObserver):
    def __init__(self, event_queue: EventQueue):
        super().__init__()
        self._commands = []
        self.event_queue = event_queue

    @property
    def commands(self):
        return self._commands
    
    @commands.setter
    def commands(self, value):
        self._commands = value
        # Adicionando a cadeia como observadora dos comandos, para atualização após finalização do execute()
        for command in self._commands:
            command.add_observers([self])
    
    @property
    def current_event(self) -> EventCommand:
        return self._current_event
    
    def update(self):
        self._current_event = self.commands.pop(0)
        self.event_queue.add(self.current_event)


class SimpleLoadNodeChain(EventChain):
    def __init__(self, event_queue: EventQueue, train: Train, node: Node, process_time) -> None:
        super().__init__(event_queue=event_queue)
        self.commands = [
            StartLoad(train=train, node=node),
            FinishLoad(train=train, node=node, process_time=process_time)
        ]
        
    