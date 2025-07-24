import sys

# Adicionando o diretório ao sys.path
sys.path.append('../')

from models.DQNRouter import DQNRouter, ActorLearner, ActionSpace
from models.tfr_state_factory import TFRStateSpaceFactory
from models.system_evolution_memory import RailroadEvolutionMemory, GlobalMemory
from models.event import DecoratedEventFactory
from models.event_calendar import EventCalendar
from models.des_simulator import DESSimulator
import dill
from datetime import datetime, timedelta
from models.clock import Clock
from tqdm import tqdm
from logging import info, critical
import warnings
from models.target import SimpleTargetManager

warnings.filterwarnings('ignore')
N_EPISODES = 10

def run_episode():
    with open('../tests/artifacts/model_to_train_15.dill', 'rb') as f:
        model = dill.load(f)
    target = SimpleTargetManager(demand=model.demands)

    local_memory = RailroadEvolutionMemory()
    event_factory = DecoratedEventFactory(
        pre_method=local_memory.save_previous_state,
        pos_method=local_memory.save_consequence
    )
    calendar = EventCalendar(event_factory=event_factory)
    sim = DESSimulator(clock=model.mesh.load_points[0].clock, calendar=calendar)
    state_space = TFRStateSpaceFactory(model)
    learner = ActorLearner(
        state_space=state_space,
        action_space=ActionSpace(model.demands),
        policy_net_path='../serialized_models/policy_net_150x6_TFRState_v1_TargetBased.dill',
        target_net_path='../serialized_models/target_net_150x6_TFRState_v1_TargetBased.dill',
    )
    global_memory = GlobalMemory()
    local_memory.add_observers([global_memory])
    global_memory.add_observers([learner])
    router = DQNRouter(
        state_space=state_space,
        demands=model.demands,
        policy_net=learner.policy_net,
        simulation_memory=local_memory,
        epsilon=1.0,
        exploration_method=target.furthest_from_the_target,
    )

    model.router = router
    local_memory.railroad = model
    local_memory.add_observers([learner])
    with learner:
        sim.simulate(model=model, time_horizon=timedelta(days=30))

    return router.operated_volume(), router.total_demand(), router.epsilon


for episode in range(N_EPISODES):
    op_vol, dem, final_epsilon = run_episode()
    critical(f'Episode {episode} - Volume: {op_vol} - Demanda: {dem} - epsilon: {final_epsilon}')
