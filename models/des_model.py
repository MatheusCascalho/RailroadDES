import abc
from interfaces.des_simulator_interface import DESSimulatorInterface
from interfaces.train_interface import TrainInterface
from models.event_calendar import Event
from models.conditions import RailroadMesh
from models.states import RailroadState
from models.inputs import Demand
from models.exceptions import TrainExceptions, FinishedTravelException
from models.constants import TrainActions


class DESModel(abc.ABC):
    def __init__(
            self,
            controllable_events: list[Event],
            uncontrollable_events: list[Event],
    ):
        self.controllable_events = []
        self.uncontrollable_events = []

    @abc.abstractmethod
    def starting_events(self, simulator: DESSimulatorInterface):
        pass

    @abc.abstractmethod
    def solver_exceptions(self, exception: Exception, event: Event):
        pass


class Railroad(DESModel):
    def __init__(self, mesh: RailroadMesh, trains: list[TrainInterface], demands: list[Demand]):
        super().__init__(
            controllable_events=[],
            uncontrollable_events=[]
        )
        self.mesh = mesh
        self.trains = trains
        self.state: RailroadState = RailroadState(
            operated_volume=0,
            completed_travels=0,
            loaded_trains=0,
            empty_trains=0
        )

    # ===== Events =========
    def starting_events(self, simulator: DESSimulatorInterface):
        for train in self.trains:
            if train.action == TrainActions.MOVING:
                origin = train.current_location[0]
                destination = train.current_location[1]
            else:
                origin = train.current_location
                try:
                    destination = train.next_location
                except TrainExceptions:
                    train.path = self.create_new_path(current_location=origin)
                    destination = train.next_location

            time = self.mesh.transit_time(origin_id=origin, destination_id=destination)

            simulator.add_event(
                time=time,
                callback=train.arrive,
                simulator=simulator,
                node=self.mesh.load_points[0],
            )

    def solver_exceptions(self, exception: Exception, event: Event):
        if isinstance(exception, FinishedTravelException):
            train: TrainInterface = exception.train
            train.path = self.create_new_path(current_location=train.current_location)


    def on_finish_loaded_path(self, simulator, train: TrainInterface):

        time = 0#simulator.time + train.state.time_register.tim
        simulator.add_event(
            time=time,
            callback=self.on_unfinish_loading,
            simulator=simulator
        )

    def on_finish_loading(self, simulator, train):
        origin = train.current_location
        destination = train.next_location
        time = simulator.current_date + self.mesh.transit_time(origin_id=origin, destination_id=destination)
        simulator.add_event(
            time=time,
            callback=self.on_finish_loaded_path,
            simulator=simulator,
            train=train
        )

    # ===== Events =========
    # ===== Decision Methods =========
    @staticmethod
    def create_new_path(current_location: int):
        return [1, 0]
