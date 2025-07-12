from models.TFRState import TFRState, TFRStateSpace
from models.tfr_state_factory import TFRStateFactory, TFRStateSpaceFactory
import dill
from models.des_simulator import DESSimulator
from models.clock import Clock
from datetime import datetime, timedelta


def simulate():
    sim = DESSimulator(clock=Clock(start=datetime(2025,4,1), discretization=timedelta(hours=1)))
    # model = create_model(sim=sim, n_trains=3)
    with open('../tests/artifacts/model_2.dill', 'rb') as f:
        model = dill.load(f)
        sim.clock = model.mesh.load_points[0].clock

    sim.simulate(model=model, time_horizon=timedelta(days=5))
    return model

model = simulate()
space = TFRStateSpaceFactory(model)
state = TFRStateFactory(model)
tensor = space.to_tensor(state)

