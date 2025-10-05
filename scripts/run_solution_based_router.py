import dill
from torch import mode
from models.event import Event
from models.event_calendar import EventCalendar
from models.des_simulator import DESSimulator
from models.solution_based_router import SolutionBasedRouter
from datetime import timedelta
from models.VNS import VNSalgorithm, SwapToFast, Move

SECONDS_IN_DAY = 60*60*24
base_model = 'tests/artifacts/model_to_train_15_sim_v2.dill'

def load_model():
    with open(base_model, 'rb') as f:
        model = dill.load(f)
        for up in model.mesh.unload_points:
            up.stocks['product'].capacity *= 100_00
    return model


def objective_function(solution):
    model = load_model()
    
    calendar = EventCalendar()
    start = model.mesh.load_points[0].clock.current_time
    sim = DESSimulator(clock=model.mesh.load_points[0].clock, calendar=calendar)

    router = SolutionBasedRouter(
        demands=model.demands,
        train_size=6e3,
        railroad_mesh=model.mesh,
        initial_solution=solution
    )

    model.router = router
    sim.simulate(model=model, time_horizon=timedelta(days=1_000_000))
    end = model.mesh.load_points[0].clock.current_time
    ellapsed_time = end - start
    f = ellapsed_time.total_seconds()/SECONDS_IN_DAY
    return f

def SimToSwap(sol):

    model = load_model()


    calendar = EventCalendar()
    sim = DESSimulator(clock=model.mesh.load_points[0].clock, calendar=calendar)
    router = SolutionBasedRouter(
        demands=model.demands,
        train_size=6e3,
        railroad_mesh=model.mesh,
        initial_solution=sol
    )
    model.router = router
    sim.simulate(model=model, time_horizon=timedelta(days=1_000_000))
    neigbor = SwapToFast(solution=router.solution, router=router)
    return neigbor

def get_initial_solution():
    model = load_model()

    router = SolutionBasedRouter(
        demands=model.demands,
        train_size=6e3,
        railroad_mesh=model.mesh,
    )
    return router.solution


neighbors = [
    SimToSwap,
    Move
]

sol = get_initial_solution()
vns = VNSalgorithm(
    solution=sol,
    neighborhoods=neighbors,
    objective_function=objective_function
)
vns.solve()
print()