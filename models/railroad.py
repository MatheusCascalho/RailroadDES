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
            self.router.route(current_time=simulator.current_date, train=train, state=self.state)

            segments = self.mesh.get_segments(train.current_task)
            scheduler = ArriveScheduler(
                rail_segments=segments,
                simulator=simulator
            )
            train.add_observers([scheduler])
            scheduler.update()

    def solver_exceptions(self, exception: Exception, event: Event, simulator: DESSimulatorInterface):
        if isinstance(exception, FinishedTravelException):
            train: TrainInterface = exception.train
            self.router.route(train=train,current_time=exception.current_time, state=self.state)
            segments = self.mesh.get_segments(train.current_task)
            scheduler = ArriveScheduler(
                rail_segments=segments,
                simulator=simulator
            )
            train.add_observers([scheduler])

    def to_json(self):
        return dict(
            mesh=self.mesh.to_json(),
            trains=len(self.trains),
            demands=[d.to_json() for d in self.demands],
        )

    # ===== Events =========
    # ===== Decision Methods =========

    def __repr__(self):
        return f"Railroad with {len(self.mesh)} nodes and {len(self.trains)} trains"
