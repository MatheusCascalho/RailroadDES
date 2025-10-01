from dataclasses import dataclass, field
from typing import List
from models.demand import Flow
from models.states import ActivityState
from torch.nn import Embedding
import torch
import numpy as np


@dataclass(frozen=True)
class TrainState:
    name: str
    local: str
    activity: ActivityState
    transit_time: float = 0
    transit_weight_punishment: float = -2
    loading_weight_reward: float = 500
    queue_weight_punishment: float = -20

    def reward(self) -> float:
        r = 0
        if self.activity == ActivityState.PROCESSING:
            r += self.loading_weight_reward
        elif self.activity == ActivityState.MOVING:
            r += self.transit_time * self.transit_weight_punishment
        else:
            r += self.queue_weight_punishment
        return r

    def __str__(self):
        return f"Trem {self.name} em {self.local}-{self.activity.value}"

@dataclass(frozen=True)
class FlowState:
    flow: Flow
    has_demand: bool
    missing_volume: float
    completed_flow_weight_reward: float = 5000

    @property
    def name(self):
        return str(self.flow)

    def reward(self) -> float:
        r = (1-self.missing_volume) * self.completed_flow_weight_reward
        return r
    
    def __str__(self):
        return f"Fluxo {self.flow.origin}-{self.flow.destination} com {round(100*(1-self.missing_volume))}% de atendimento"


@dataclass(frozen=True)
class ConstraintState:
    name: str
    is_blocked: bool
    blocked_constraint_weight: float = -10

    def reward(self) -> float:
        r = self.blocked_constraint_weight if self.is_blocked else 0
        return r


@dataclass(frozen=True)
class TFRState:
    train_states: List[TrainState]
    constraint_states: List[ConstraintState]
    flow_states: List[FlowState]
    is_initial: bool

    def __post_init__(self):
        pass

    def reward(self) -> float:
        r = 0
        for ts in self.train_states:
            r += ts.reward()
        for cs in self.constraint_states:
            r += cs.reward()
        for fs in self.flow_states:
            r += fs.reward()
        return r / 1e5

    def detailed_reward(self):
        map = {ts.name: ts.reward() for ts in self.train_states}
        map.update({cs.name: cs.reward() for cs in self.constraint_states})
        map.update({fs.name: fs.reward() for fs in self.flow_states})
        return map


    @property
    def is_final(self):
        return all([not f.has_demand for f in self.flow_states])
    
    def __str__(self):
        s = ""
        for f in self.flow_states:
            s += str(f) + "\n"
        for t in self.train_states:
            s += str(t) + "\n"
        for c in self.constraint_states:
            if c.is_blocked:
                s += str(c.name) + "\n"
        s += f"Recompensa {round(self.reward())}"
        return s

class TFRBalanceState(TFRState):
    penalty_factor: float = 1e3
    def reward(self) -> float:
        base_reward = sum([1e5 if f.activity == ActivityState.PROCESSING else 0 for f in self.train_states])
        missing_volumes = [f.missing_volume for f in self.flow_states]
        diff = sum(np.abs(np.diff(missing_volumes)))

        # std_volumes = np.std(missing_volumes)
        penalty = diff * self.penalty_factor
        reward = base_reward - penalty
        return reward # type: ignore


EMBEDDING_SPACE_DIMENSION = 5

class TFRStateSpace:
    def __init__(
            self,
            mesh_edges: list[tuple[str, str]],
            activities: list[str],
            train_names: list[str],
            flows: list[Flow],
            constraints: list[str]
    ):
        # Attributes coded with embedding
        stations = sorted(set({s[0] for s in mesh_edges}.union({s[1] for s in mesh_edges}))) + ['origin', 'destination']
        connections = [f"_-{s}" for s in stations] 
        connections += [f"{c[0]}-{c[1]}" for c in mesh_edges]
        connections += [f"{c[1]}-{c[0]}" for c in mesh_edges]
        connections = list(set(connections))
        self.locals_to_index = {
            loc: i
            for i, loc in enumerate(sorted(stations+connections))}
        self.embedding_locals = Embedding(num_embeddings=len(self.locals_to_index), embedding_dim=EMBEDDING_SPACE_DIMENSION)

        # Attributes coded with one-hot encoding
        a = len(activities)
        activities_encoded = np.eye(a)
        self.activities = {a:r for a, r in zip(activities, activities_encoded)}

        f = len(flows)
        flows_encoded = np.eye(f)
        self.flows = {f: r for f, r in zip(flows, flows_encoded)}

        # Cardinalities
        self.train_names = sorted(train_names)
        self.constraints = sorted(constraints)

    @property
    def cardinality(self):
        n = 0

        # cardinalidade dos locais
        trains = len(self.train_names)
        n += trains * EMBEDDING_SPACE_DIMENSION

        # cardinalidade das atividades
        acitivities = len(self.activities)
        n += trains * acitivities

        # cardinalidade dos fluxos
        flows = len(self.flows)
        n += flows

        # cardinalidade das restrições
        constraints = len(self.constraints)
        n += constraints

        return n

    def to_array(self, state: TFRState):
        locations = []
        activities = []
        flows = []
        constraints = []

        for t in state.train_states:
            idx = torch.tensor([self.locals_to_index[t.local]])
            try:
                loc = self.embedding_locals(idx)
            except:
                loc = [0,0,0,0]
            loc = loc.detach().numpy()[0].tolist()
            locations += loc

            a = self.activities[t.activity.name]

            activities.append(a)
        activities = list(np.array(activities).flatten())

        for flow in state.flow_states:
            flows.append(flow.missing_volume)

        for constraint in state.constraint_states:
            constraints.append(int(constraint.is_blocked))

        locations = [l for l in locations]
        arr = locations+activities+flows+constraints
        arr = np.array(arr)
        return arr

