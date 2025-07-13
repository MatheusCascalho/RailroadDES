import sys

# Adicionando o diretório ao sys.path
sys.path.append('../')

from models.DQNRouter import DQNRouter
from models.tfr_state_factory import TFRStateSpaceFactory
from models.system_evolution_memory import RailroadEvolutionMemory
from models.event import DecoratedEventFactory
from models.event_calendar import EventCalendar
from models.des_simulator import DESSimulator
import dill
from datetime import datetime, timedelta
from models.clock import Clock
from tqdm import tqdm
from logging import info, critical
import warnings
warnings.filterwarnings('ignore')
N_EPISODES = 10000

def run_episode():
    with open('../tests/artifacts/model_to_train.dill', 'rb') as f:
        model = dill.load(f)

    memory = RailroadEvolutionMemory()
    event_factory = DecoratedEventFactory(
        pre_method=memory.save_previous_state,
        pos_method=memory.save_consequence
    )
    calendar = EventCalendar(event_factory=event_factory)
    sim = DESSimulator(clock=model.mesh.load_points[0].clock, calendar=calendar)
    state_space = TFRStateSpaceFactory(model)
    router = DQNRouter(state_space=state_space, demands=model.demands, epsilon=1.0)

    model.router = router
    memory.railroad = model
    memory.add_observers([router])
    with router:
        sim.simulate(model=model, time_horizon=timedelta(days=30))

    return router.operated_volume(), router.total_demand()


for episode in range(N_EPISODES):
    op_vol, dem = run_episode()
    critical(f'Episode {episode} - Volume: {op_vol} | Demanda: {dem}')
