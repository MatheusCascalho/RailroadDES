from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
from interfaces.train_interface import TrainInterface


@dataclass
class QueueElement:
    element: TrainInterface
    arrive: datetime
    departure: datetime = field(init=False, default=None)


class Queue:
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.elements: list[QueueElement] = []
        self.history: list[QueueElement] = []

    def clear(self):
        ...

    def pop(self, current_time):
        data = self.elements.pop(0)
        data.departure = current_time
        self.history.append(data)
        return data.element

    def push(self, element, arrive):
        if len(self.elements) < self.capacity - 1:
            queue_element = QueueElement(element=element, arrive=arrive)
            self.elements.append(queue_element)
        else:
            raise Exception('Queue is completely full!!')

    def statistics(self):
        ...

    @property
    def current_size(self):
        return len(self.elements)

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
