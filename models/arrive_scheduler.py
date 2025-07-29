from interfaces.des_simulator_interface import DESSimulatorInterface
from interfaces.train_interface import TrainInterface
from models.railroad_mesh import RailSegment
from models.observers import AbstractObserver
from models.exceptions import FinishedTravelException


class ArriveScheduler(AbstractObserver):
    def __init__(
            self,
            rail_segments: list[RailSegment],
            simulator: DESSimulatorInterface,
            train: TrainInterface = None
    ):
        self.rail_segments = rail_segments
        self.simulator = simulator
        super().__init__()

    def append_subject(self, sub, update=False):
        if len(self.subjects)>0:
            raise Exception("Scheduler is already looking to a train")
        if not isinstance(sub, TrainInterface):
            raise Exception("Scheduler only look for Train Objects")
        self.subjects.append(sub)
        if update:
            self.update()

    def send(self):
        current_segment = self.current_segment()

        self.simulator.add_event(
            time=current_segment.time_to_destination,
            callback=current_segment.destination.receive,
            train=self.subjects[0],
            simulator=self.simulator,
        )

    def update(self):
        if self.subjects[0].dispatched_just_now:
            self.send()
        if self.subjects[0].arrived_right_now:
            self.rail_segments = self.rail_segments[1:]

    def current_segment(self):
        if not self.rail_segments:
            FinishedTravelException.path_is_finished(
                train=self.subjects[0],
                current_time=self.simulator.current_date
            )
        return self.rail_segments[0]
