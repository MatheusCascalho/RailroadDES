from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
from interfaces.train_interface import TrainInterface


@dataclass
class QueueElement:
    element: TrainInterface
    arrive: datetime
    departure: datetime = field(init=False, default=None)

    def __str__(self):
        return self.element.ID
    __repr__=__str__


class Queue:
    def __init__(self, capacity: int, name: str):
        self.name = name
        self.capacity: int = capacity
        self.elements: list[QueueElement] = []
        self.skipped: list[QueueElement] = []
        self.history: list[QueueElement] = []

    def __str__(self):
        return f"Queue {self.name} with {self.current_size} elements"

    __repr__ = __str__

    def has_next(self):
        return len(self.elements)>0

    def running_queue(self):
        while self.has_next():
            yield self.elements[0]
        raise StopIteration

    @property
    def first(self):
        if not self.elements:
            return None
        return self.elements[0].element


    def skip_process(self, process):
        self.skipped = [e for e in self.elements if e.element.current_process_name == process]
        self.elements = [e for e in self.elements if e.element.current_process_name != process]

    def recover(self):
        self.elements += self.skipped
        self.skipped = []

    def clear(self):
        ...

    @property
    def is_full(self):
        return self.current_size == self.capacity

    @property
    def is_busy(self):
        return self.current_size > 0

    def pop(self, current_time):
        data = self.elements.pop(0)
        data.departure = current_time
        self.history.append(data)
        return data.element

    def push(self, element, arrive):
        if len(self.elements) < self.capacity:
            queue_element = QueueElement(element=element, arrive=arrive)
            self.elements.append(queue_element)
        else:
            raise Exception('Queue is completely full!!')

    def statistics(self):
        ...

    @property
    def current_size(self):
        return len(self.elements) + len(self.skipped)

    def now(self):
        return [e.element for e in self.elements]

    def __iter__(self):
        return self.elements.__iter__()

    def __str__(self):
        return f"Queue with {self.current_size} elements. Available space: {self.capacity - self.current_size}"


if __name__ == '__main__':
    model_queue = Queue(capacity=5)
    print(model_queue)
    model_queue.push('5', arrive=datetime(2020, 1, 1))
    print(model_queue)
    model_queue.pop(current_time=datetime(2020, 9, 1))
    print(model_queue)
    print(model_queue.history)
