from datetime import datetime

import pandas as pd

from interfaces.des_simulator_interface import DESSimulatorInterface
from interfaces.train_interface import TrainInterface
from models.arrive_scheduler import ArriveScheduler
from models.demand import Demand
from models.des_model import DESModel
from models.event_calendar import Event
from models.exceptions import FinishedTravelException
from models.railroad_mesh import RailroadMesh
from models.router import Router


class Railroad(DESModel):
    def __init__(
            self,
            mesh: RailroadMesh,
            trains: list[TrainInterface],
            demands: list[Demand],
            router: Router
    ):
        super().__init__(
            controllable_events=[],
            uncontrollable_events=[]
        )
        self.mesh = mesh
        self.trains = trains
        self.demands = demands
        self.router = router
        # self.petri_model = self.build_petri_model()

    # ===== Events =========
    @property
    def state(self):
        nodes = '\n'.join([f"{n} - {n.state}" for n in self.mesh])
        trains = '\n'.join([f"{t} - {t.state}" for t in self.trains])
        return f"{nodes}\n{trains}"

    def starting_events(self, simulator: DESSimulatorInterface):
        for train in self.trains:
            self.router.route(current_time=simulator.current_date, train=train)

            segments = self.get_segments(train.current_task)
            scheduler = ArriveScheduler(
                rail_segments=segments,
                simulator=simulator
            )
            train.add_observers([scheduler])
            scheduler.update()

    def get_segments(self, task):
        segments = []
        last = ''
        for n in task.path.path:
            if '-' not in n:
                continue
            o, d = n.split('-')
            if o in self.mesh.graph:
                s = self.mesh.graph[o][0]
            else:
                s = self.mesh.graph[d][0].reversed()
            segments.append(s)
        return segments


    def solver_exceptions(self, exception: Exception, event: Event, simulator: DESSimulatorInterface):
        if isinstance(exception, FinishedTravelException):
            train: TrainInterface = exception.train
            self.router.route(train=train,current_time=exception.current_time)
            segments = self.get_segments(train.current_task)
            scheduler = ArriveScheduler(
                rail_segments=segments,
                simulator=simulator
            )
            train.add_observers([scheduler])

    def stop_train(self, **kwargs):
        pass
    # ===== Events =========
    # ===== Decision Methods =========
    def create_new_path(self, current_time: datetime, current_location):
        path, demand = self.choose_path(current_time=current_time, current_location=current_location)
        return path, demand

    def choose_path(self, current_time, current_location):
        paths = []
        for demand in self.demands:
            if not demand.is_completed:
                path = self.mesh.complete_path(
                    origin_name=demand.origin, destination_name=demand.destination
                )
                predicted_time = self.mesh.predicted_time_for_path(path=path, current_time=current_time)
                paths.append((predicted_time, path, demand))

        paths = sorted(paths, key=lambda x: x[0])
        choosed_path = paths[0][1]
        demand = paths[0][2]
        return choosed_path, demand

    def statistics(self):
        operated_volume = [
            {"Origin": demand.flow.origin, "Destination": demand.flow.destination, "Demand": demand.volume,
             "Operated": demand.operated, "Cut": demand.cut}
            for demand in self.demands
        ]
        return pd.DataFrame(operated_volume)

    def __repr__(self):
        return f"Railroad with {len(self.mesh)} nodes and {len(self.trains)} trains"
