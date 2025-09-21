from models.railroad import Railroad
from models.TFRState import TFRState, TrainState, ConstraintState, FlowState, TFRStateSpace
from models.states import ActivityState

def TFRStateFactory(railroad: Railroad, is_initial=False, tfr_class = TFRState) -> TFRState:
        trains = [
            TrainState(
                name=t.ID,
                local=t.current_location,
                activity=t.current_activity.name,
                transit_time=railroad.get_transit_time(t)
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
                has_demand=not d.is_completed,
                missing_volume=d.cut/d.volume,
                completed_flow_weight_reward=d.volume
            )
            for d in railroad.router.demands
        ]

        state = tfr_class(
            train_states=trains,
            constraint_states=restrictions,
            flow_states=flows,
            is_initial=is_initial
        )
        return state

def TFRStateSpaceFactory(railroad: Railroad) -> TFRStateSpace:
    space = TFRStateSpace(
        mesh_edges=[
            (t.origin.name, t.destination.name) 
            for _, segments in railroad.mesh.graph.items()
            for t in segments
        ],
        activities=[a.name for a in ActivityState],
        train_names=[t.ID for t in railroad.trains],
        flows=[d.flow for d in railroad.router.demands],
        constraints=[f"{n.name} | {c.ID}" for n in railroad.mesh for c in n.constraints],
    )
    return space