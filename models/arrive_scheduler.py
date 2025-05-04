from interfaces.des_simulator_interface import DESSimulatorInterface
from interfaces.train_interface import TrainInterface
from models.railroad import RailSegment
from models.observers import AbstractObserver
from models.time_table import TimeTable
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
        self.train = train
        super().__init__()

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value

    def append_subject(self, sub):
        if not isinstance(sub, TimeTable):
            raise Exception("Scheduler only look for TimeTable Objects")
        self.subjects.append(sub)
        self.update()

    def send(self):
        try:
            current_segment = self.current_segment()
        except FinishedTravelException:
            self.simulator.solve_exceptions()
            current_segment = self.current_segment()

        self.simulator.add_event(
            time=current_segment.time_to_destination,
            callback=current_segment.destination.receive,
            train=self.train
        )

    def update(self):
        if self.train and self.subjects[0].dispatched_just_now:
            self.send()
        if self.subjects[0].arrived_right_now:
            self.rail_segments = self.rail_segments[1:]

    def current_segment(self):
        if not self.rail_segments:
            FinishedTravelException.path_is_finished(
                train=self.train,
                current_time=self.simulator.current_date
            )
        return self.rail_segments[0]
