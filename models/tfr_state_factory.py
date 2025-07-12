from models.railroad import Railroad
from models.TFRState import TFRState, TrainState, ConstraintState, FlowState, TFRStateSpace
from models.states import ActivityState

def TFRStateFactory(railroad: Railroad) -> TFRState:
        trains = [
            TrainState(
                name=t.ID,
                local=t.current_location,
                activity=t.current_activity,
            )
            for t in railroad.trains
        ]

        restrictions = [
            ConstraintState(
                name=c.ID,
                is_blocked=c.is_blocked(),
            )
            for n in railroad.mesh
            for c in n.constraints
        ]

        flows = [
            FlowState(
                flow=d.flow,
                has_demand=not d.is_completed
            )
            for d in railroad.router.demands
        ]

        state = TFRState(
            train_states=trains,
            constraint_states=restrictions,
            flow_states=flows,
        )
        return state

def TFRStateSpaceFactory(railroad: Railroad) -> TFRStateSpace:
    space = TFRStateSpace(
        mesh_edges=[(t.load_origin, t.load_destination) for t in railroad.mesh.transit_times],
        activities=[a.name for a in ActivityState],
        train_names=[t.ID for t in railroad.trains],
        flows=[d.flow for d in railroad.router.demands],
        constraints=[f"{n.name} | {c.ID}" for n in railroad.mesh for c in n.constraints],
    )
    return space