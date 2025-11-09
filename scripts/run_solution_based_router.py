import dill
from torch import mode
from src.control.memory_based_control.reinforcement_learning.DQNRouter import Learner
from src.control.memory_based_control.reinforcement_learning.action_space import ActionSpace
from src.events.event import DecoratedEventFactory, Event
from src.simulation.event_calendar import EventCalendar
from src.simulation.des_simulator import DESSimulator
from src.control.solution_based_control.solution_based_router import SolutionBasedRouter
from datetime import timedelta
from src.optimization.VNS import VNSalgorithm, SwapToFast, Move
from src.control.memory_based_control.system_evolution_memory import RailroadEvolutionMemory
from src.control.memory_based_control.reinforcement_learning.tfr_state_factory import TFRStateFactory, TFRStateSpaceFactory

SECONDS_IN_DAY = 60*60*24
base_model = 'tests/artifacts/model_to_train_15_sim_v2.dill'

policy_net_path='serialized_models/dqn/policy_net_15_trains_10out25_VNS_2.pytorch'
target_net_path='serialized_models/dqn/target_net_15_trains_10out25_VNS_2.pytorch'

def load_model():
    with open(base_model, 'rb') as f:
        model = dill.load(f)
        for up in model.mesh.unload_points:
            up.stocks['product'].capacity *= 100_00
    return model


def objective_function(solution):
    model = load_model()
    
    state_space = TFRStateSpaceFactory(model)
    def state_factory_wrapper(**kwargs):
        state = TFRStateFactory(**kwargs)
        return state
    local_memory = RailroadEvolutionMemory(state_factory=state_factory_wrapper)
    local_memory.railroad = model

    event_factory = DecoratedEventFactory(
        pre_method=local_memory.save_previous_state,
        pos_method=local_memory.save_consequence
    )
    calendar = EventCalendar(event_factory=event_factory)

    learner = Learner(
        state_space=state_space,
        action_space=ActionSpace(model.demands),
        policy_net_path=policy_net_path,
        target_net_path=target_net_path,
        epsilon_decay_steps=100,
        epsilon_start=0.9,
        epsilon_end=0,
        batch_size=15,
        target_update_freq=1000
    )
    local_memory.add_observers([learner])

    start = model.mesh.load_points[0].clock.current_time
    sim = DESSimulator(clock=model.mesh.load_points[0].clock, calendar=calendar)

    router = SolutionBasedRouter(
        demands=model.demands,
        train_size=6e3,
        railroad_mesh=model.mesh,
        initial_solution=solution
    )

    model.router = router
    sim.simulate(model=model, time_horizon=timedelta(days=90), starting_time_horizon=timedelta(days=90))
    learner.save()
    end = model.mesh.load_points[0].clock.current_time
    ellapsed_time = end - start
    f = ellapsed_time.total_seconds()/SECONDS_IN_DAY
    return f

def SimToSwap(sol, k):
    model = load_model()
    state_space = TFRStateSpaceFactory(model)
    def state_factory_wrapper(**kwargs):
        state = TFRStateFactory(**kwargs)
        return state
    local_memory = RailroadEvolutionMemory(state_factory=state_factory_wrapper)
    local_memory.railroad = model

    event_factory = DecoratedEventFactory(
        pre_method=local_memory.save_previous_state,
        pos_method=local_memory.save_consequence
    )
    calendar = EventCalendar(event_factory=event_factory)

    sim = DESSimulator(clock=model.mesh.load_points[0].clock, calendar=calendar)
    router = SolutionBasedRouter(
        demands=model.demands,
        train_size=6e3,
        railroad_mesh=model.mesh,
        initial_solution=sol
    )
    learner = Learner(
        state_space=state_space,
        action_space=ActionSpace(model.demands),
        policy_net_path=policy_net_path,
        target_net_path=target_net_path,
        epsilon_decay_steps=100,
        epsilon_start=0.9,
        epsilon_end=0,
        batch_size=15,
        target_update_freq=1000
    )
    local_memory.add_observers([learner])
    model.router = router
    sim.simulate(model=model, time_horizon=timedelta(days=1_000_000))
    learner.save()
    neigbor = SwapToFast(solution=router.solution, router=router, learner=learner, k=k)
    return neigbor

def get_initial_solution():
    model = load_model()

    router = SolutionBasedRouter(
        demands=model.demands,
        train_size=6e3,
        railroad_mesh=model.mesh,
    )
    return router.solution

def SimToSwapDQN():
    def wrapper(sol):
        return SimToSwap(sol, k='DQN')
    return wrapper

def SimToSwapCycle():
    def wrapper(sol):
        return SimToSwap(sol, k='Cycle')
    return wrapper


neighbors = [
    SimToSwapDQN(),
    Move,
    SimToSwapCycle(),
]

sol = get_initial_solution()
vns = VNSalgorithm(
    solution=sol,
    neighborhoods=neighbors,
    objective_function=objective_function
)
vns.solve()
print()