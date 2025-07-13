from datetime import datetime

import pandas as pd
from datetime import timedelta
from interfaces.des_simulator_interface import DESSimulatorInterface
from interfaces.train_interface import TrainInterface
from models.arrive_scheduler import ArriveScheduler
from models.demand import Demand
from models.des_model import DESModel
from models.event import Event
from models.exceptions import FinishedTravelException
from models.railroad_mesh import RailroadMesh
from models.router import Router, RandomRouter
from models.states import ActivityState

SECONDS_IN_HOUR = 60*60


class Railroad(DESModel):
    def __init__(
            self,
            mesh: RailroadMesh,
            trains: list[TrainInterface],
            demands: list[Demand],
            router: Router = None
    ):
        super().__init__(
            controllable_events=[],
            uncontrollable_events=[]
        )
        self.mesh = mesh
        self.trains = trains
        self.demands = demands
        self._router = router
        # self.petri_model = self.build_petri_model()

    # ===== Events =========
    @property
    def router(self):
        if self._router is None:
            self._router = RandomRouter(demands=self.demands)
        return self._router

    @router.setter
    def router(self, value):
        self._router = value

    @property
    def state(self):
        nodes = '\n'.join([f"{n} - {n.state}" for n in self.mesh])
        trains = '\n'.join([f"{t} - {t.state}" for t in self.trains])
        demand_trains = [(k.demand.flow, t, t.load_system.state_machine.current_state.name.name) for k, t in self.router.running_tasks.items()]
        trains_distribution = {d.flow: {s.name: 0 for s in self.trains[0].load_system.state_machine.states} for d in self.router.demands}
        for dt in demand_trains:
            trains_distribution[dt[0]][dt[2]] += 1

        trains_distribution = '====\n' + '\n'.join(f'{k}: {v}' for k, v in trains_distribution.items()) + '\n===='
        return f"{nodes}\n{str(trains_distribution)}\n{trains}"

    def starting_events(self, simulator: DESSimulatorInterface, time_horizon: timedelta):
        for train in self.trains:
            self.router.route(current_time=simulator.current_date, train=train, state=self.state)

            segments = self.mesh.get_segments(train.current_task)
            scheduler = ArriveScheduler(
                rail_segments=segments,
                simulator=simulator
            )
            train.add_observers([scheduler])
            scheduler.update()

        for node in self.mesh:
            for i in range(1, int(time_horizon.total_seconds()/(60*60*24))+1):
                simulator.add_event(
                    time=i*timedelta(days=1),
                    callback=node.pre_processing,
                )
                node.process(simulator=simulator)

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

    def get_transit_time(self, t: TrainInterface):
        if t.current_activity.name != ActivityState.MOVING:
            return 0
        segment = self.mesh.get_current_segment(t.current_task)
        time = segment.time_to_destination
        time = time.total_seconds() / SECONDS_IN_HOUR
        return time


    # ===== Events =========
    # ===== Decision Methods =========

    def __repr__(self):
        return f"Railroad with {len(self.mesh)} nodes and {len(self.trains)} trains"
