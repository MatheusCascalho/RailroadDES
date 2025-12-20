from src.events.event import EventCommand
from src.standards.observers import AbstractObserver, AbstractSubject, to_notify

class EventQueue(AbstractSubject):
    def __init__(self):
        super().__init__()
        self.queue = []
        self.chains = []

    @to_notify()
    def add(self, value):
        self.queue.append(value)

    def pop(self) -> EventCommand:
        return self.queue.pop(0)