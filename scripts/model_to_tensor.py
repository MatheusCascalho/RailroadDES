from src.TFRState import TFRState, TFRStateSpace, TFRBalanceState
from src.tfr_state_factory import TFRStateFactory, TFRStateSpaceFactory
import dill
from src.des_simulator import DESSimulator
from src.clock import Clock
from datetime import datetime, timedelta

base_model = 'tests/artifacts/simple_model_to_train_1_sim_v2.dill'

def simulate():
    sim = DESSimulator(clock=Clock(start=datetime(2025,4,1), discretization=timedelta(hours=1)))
    # model = create_model(sim=sim, n_trains=3)
    with open(base_model, 'rb') as f:
        model = dill.load(f)
        sim.clock = model.mesh.load_points[0].clock

    sim.simulate(model=model, time_horizon=timedelta(days=5))
    return model

model = simulate()
space = TFRStateSpaceFactory(model)
state = TFRStateFactory(model, tfr_class=TFRBalanceState)
reward = state.reward()
...
