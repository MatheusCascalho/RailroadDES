from interfaces.node_interce import NodeInterface
from datetime import timedelta
from interfaces.train_interface import TrainInterface

class RailSegment:
    def __init__(
            self,
            origin: NodeInterface,
            destination: NodeInterface,
            time_to_origin: timedelta,
            time_to_destination: timedelta
    ):
        self.origin = origin
        self.destination = destination
        self.to_origin = []
        self.time_to_origin = time_to_origin
        self.to_destination = []
        self.time_to_destination = time_to_destination

    def reversed(self):
        segment = RailSegment(
            origin=self.destination,
            time_to_origin=self.time_to_destination,
            destination=self.origin,
            time_to_destination=self.time_to_origin
        )
        return segment

    def send(self, train: TrainInterface):
        next_location = train.next_location
        if next_location == self.origin:
            self.to_origin.append(train)
        else:
            self.to_destination.append(train)

    def __repr__(self):
        return f"{self.origin} to {self.destination}"

