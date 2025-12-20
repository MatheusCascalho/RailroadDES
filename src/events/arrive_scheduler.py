from interfaces.des_simulator_interface import DESSimulatorInterface
from interfaces.train_interface import TrainInterface
from src.domain.entities.railroad_mesh import RailSegment
from src.standards.observers import AbstractObserver
from src.domain.exceptions import FinishedTravelException


class ArriveScheduler(AbstractObserver):
    def __init__(
            self,
            rail_segments: list[RailSegment],
            simulator: DESSimulatorInterface,
            train: TrainInterface | None = None
    ):
        self.rail_segments = rail_segments
        self.simulator = simulator
        super().__init__()

    def append_subject(self, sub: TrainInterface, update=False):
        if len(self.subjects)>0:
            raise Exception("Scheduler is already looking to a train")
        if not isinstance(sub, TrainInterface):
            raise Exception("Scheduler only look for Train Objects")
        self.subjects.append(sub) # type: ignore
        if update:
            self.update()

    @property
    def observed_train(self) -> TrainInterface:
        return self.subjects[0] # type: ignore

    def send(self):
        current_segment = self.current_segment()

        self.simulator.add_event(
            time=current_segment.time_to_destination,
            callback=current_segment.destination.receive,
            train=self.observed_train,
            simulator=self.simulator,
        )

    def update(self):
        if self.observed_train.dispatched_just_now:
            self.send()
        if self.observed_train.arrived_right_now:
            self.rail_segments = self.rail_segments[1:]

    def current_segment(self):
        if not self.rail_segments:
            FinishedTravelException.path_is_finished(
                train=self.observed_train,
                current_time=self.simulator.current_date
            )
        return self.rail_segments[0]
