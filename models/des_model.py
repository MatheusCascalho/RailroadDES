import abc
import pandas as pd
from interfaces.des_simulator_interface import DESSimulatorInterface
from interfaces.train_interface import TrainInterface
from models.event_calendar import Event
from models.railroad_mesh import RailroadMesh
from models.states import RailroadState
from models.demand import Demand
from models.exceptions import TrainExceptions, FinishedTravelException
from models.constants import TrainActions
from datetime import datetime
from petri_nets.petri_components import Place, Transition, Arc
from petri_nets.net import PetriNet
from models.arrive_scheduler import ArriveScheduler
from models.path import Path
from models.task import Task
from models.router import Router


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

    def build_petri_model(self):
        # demand places
        arcs = []
        places = []
        transitions = []

        for demand in self.demands:
            origin = self.mesh.name_to_node[demand.flow.origin]
            destination = self.mesh.name_to_node[demand.flow.destination]

            demand_to_be_attended = Place(
                tokens=demand.volume / self.trains[0].capacity,
                meaning=f"Demand from {demand.origin} to {demand.destination}",
                identifier=f"p_demand_{demand.origin}_{demand.destination}_to_be_attended",
            )
            demand_being_attended = Place(
                tokens=0,
                meaning=f"Demand being attended between {demand.origin} and {demand.destination}",
                identifier=f"p_demand_{demand.origin}_{demand.destination}_being_attended",
            )
            places.extend([demand_to_be_attended, demand_being_attended])
            transitions.extend([
                origin.transitions['receive'],
                destination.transitions['receive'],
            ])
            flow_arcs = [
                Arc(input=demand_to_be_attended, output=origin.transitions['receive']),
                Arc(input=demand_being_attended, output=destination.transitions['receive']),
                Arc(input=origin.transitions['receive'], output=demand_being_attended)
            ]
            arcs.extend(flow_arcs)

        loaded_trains = Place(
            tokens=self.state.loaded_trains,
            meaning="loaded trains on railroad",
            identifier="p_loaded_trains"
        )

        empty_trains = Place(
            tokens=self.state.empty_trains,
            meaning="empty trains on railroad",
            identifier="p_empty_trains"
        )
        places.extend([loaded_trains, empty_trains])

        for unload in self.mesh.unload_points:
            arcs.extend([
                Arc(input=loaded_trains, output=unload.transitions['receive']),
                Arc(input=unload.transitions['dispatch'], output=empty_trains)
            ])
            transitions.extend([
                unload.transitions['receive'],
                unload.transitions['dispatch']
            ])
        for load in self.mesh.load_points:
            arcs.extend([
                Arc(input=empty_trains, output=load.transitions['receive']),
                Arc(input=load.transitions['dispatch'], output=loaded_trains)
            ])
            transitions.extend([
                load.transitions['receive'],
                load.transitions['dispatch']
            ])

        net = PetriNet(
            places=places,
            transitions=transitions,
            arcs=arcs,
            name="railroad"
        )
        node_models = [n.petri_model for n in self.mesh.load_points+self.mesh.unload_points]
        complete_model = net.modular_composition(node_models)
        return complete_model
