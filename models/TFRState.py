from dataclasses import dataclass
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


@dataclass(frozen=True)
class FlowState:
    flow: Flow
    has_demand: bool


@dataclass(frozen=True)
class ConstraintState:
    name: str
    is_blocked: bool


@dataclass(frozen=True)
class TFRState:
    train_states: List[TrainState]
    constraint_states: List[ConstraintState]
    flow_states: List[FlowState]

    def reward(self) -> float:
        pass


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
        stations = sorted(set({s[0] for s in mesh_edges}.union({s[1] for s in mesh_edges})))
        connections = [f"_-{s}" for s in stations] + [f"{c[0]}-{c[1]}" for c in mesh_edges]

        self.locals_to_index = {
            loc: i
            for i, loc in enumerate(sorted(stations+connections))}
        self.embedding_locals = Embedding(num_embeddings=len(self.locals_to_index), embedding_dim=5)

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


    def to_tensor(self, state: TFRState):
        locations = []
        activities = []
        flows = []
        constraints = []

        for t in state.train_states:
            idx = torch.tensor([self.locals_to_index[t.local]])
            loc = self.embedding_locals(idx)
            locations.append(loc)

            a = self.activities[t.activity.name.name]

            activities.append(a)
        activities = list(np.array(activities).flatten())

        for flow in state.flow_states:
            flows.append(int(flow.has_demand))

        for constraint in state.constraint_states:
            constraints.append(int(constraint.is_blocked))
        t1 = torch.cat(locations, dim=1)
        t2 = np.array(activities+flows+constraints)
        t2 = torch.tensor(t2.reshape(1,len(t2)))
        tensor = torch.cat((t1, t2), dim=1)
        return tensor