import sys
from concurrent.futures.process import ProcessPoolExecutor

# Adicionando o diretório ao sys.path
sys.path.append('')
sys.path.append('.')
import random
from models.QLearningRouter import QRouter, QTable
from models.action_space import ActionSpace
from models.tfr_state_factory import TFRStateSpaceFactory
from models.system_evolution_memory import RailroadEvolutionMemory, ExperienceProducer
from models.event import DecoratedEventFactory
from models.event_calendar import EventCalendar
from models.des_simulator import DESSimulator
import dill
from datetime import datetime, timedelta
from models.clock import Clock
from tqdm import tqdm
from logging import info, critical, error
import warnings
from models.target import SimpleTargetManager
from multiprocessing import Event, Process, Pool, Manager
import multiprocessing
from models.dill_queue import DillQueue
import os
from time import sleep
from dataclasses import dataclass
import cProfile

warnings.filterwarnings('ignore')
# N_EPISODES = 10
EPISODES_BY_PROCESS = 1_000
NUM_PROCESSES = 1
TRAINING_STEPS = 1

@dataclass
class OutputData:
    operated_volume: float
    total_demand: float
    process_id: int
    episode_number: int
    q_table_size: int

def setup_shared_components(experience_queue):
    with open('tests/artifacts/model_to_train_15.dill', 'rb') as f:
    # with open('tests/artifacts/simple_model_to_train_1_sim_v2.dill', 'rb') as f:
        model = dill.load(f)
    state_space = TFRStateSpaceFactory(model)
    learner = QTable(
        action_space=ActionSpace(model.demands),
        q_table_file=f"q_table_10_nos_15_trens.dill"
        # q_table_file=f"q_table_pid_{os.getpid()}.dill"
    )
    global_memory = ExperienceProducer(queue=experience_queue)
    return learner, global_memory

def profile(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        r = func(*args, **kwargs)
        profiler.disable()
        profiler.dump_stats(f"{func.__name__}_PID{os.getpid()}.prof")
        return r
    return wrapper

# @profile
def learning_loop(queue, stop_event):
    with open('tests/artifacts/model_to_train_15.dill', 'rb') as f:
    # with open('tests/artifacts/simple_model_to_train_1_sim_v2.dill', 'rb') as f:
        model = dill.load(f)
    state_space = TFRStateSpaceFactory(model)
    learner = QTable(
        action_space=ActionSpace(model.demands),
        q_table_file=f"q_table_10_nos_15_trens.dill"
        # q_table_file=f"q_table_pid_{os.getpid()}.dill"    
    )
    # with learner:
    while not stop_event.is_set():
        try:
            experience = queue.get(timeout=1)
            learner.update(experience)
        except Exception as e:
            info(f'Experience queue is empty - {e}')
            # sleep(.1)
            continue
    # while not queue.empty():
    #     experience = dill.loads(queue.get(timeout=1))
    #     learner.update(experience)
    # learner.save()

# @profile
def logging_loop(stop_event, output_queue):
    log_number = 0
    while not stop_event.is_set():
        try:
            output = output_queue.get(timeout=1)
            error(f'Log {log_number} - Episode {output.episode_number} - PID: {output.process_id} - Volume: {output.operated_volume} - Demanda: {output.total_demand} - q_table_size: {output.q_table_size}')
            log_number += 1
        except Exception as e:
            info('Episode queue is empty')
            # sleep(.1)
            continue

def run_episode(episode_number, output_queue: DillQueue):
    # with open('tests/artifacts/simple_model_to_train_1_sim_v2.dill', 'rb') as f:
    with open('tests/artifacts/model_to_train_15_sim_v2.dill', 'rb') as f:
        model = dill.load(f)
    target = SimpleTargetManager(demand=model.demands)
    state_space = TFRStateSpaceFactory(model)
    learner = QTable(
        action_space=ActionSpace(model.demands),
        q_table_file=f"q_table_10_nos_15_trens.dill"
        # q_table_file=f"q_table_pid_{os.getpid()}.dill" .
    )
    experience_producer = ExperienceProducer(queue=experience_queue)

    local_memory = RailroadEvolutionMemory()
    event_factory = DecoratedEventFactory(
        pre_method=local_memory.save_previous_state,
        pos_method=local_memory.save_consequence
    )
    calendar = EventCalendar(event_factory=event_factory)
    sim = DESSimulator(clock=model.mesh.load_points[0].clock, calendar=calendar)
    local_memory.add_observers([experience_producer, learner])
    action_space = ActionSpace(model.demands)
    def always_the_same():
        yield model.demands[1]
        i = random.randint(0,1)
        yield model.demands[i]

        while True:
            yield action_space.sample()
    alws = always_the_same()
    router = QRouter(
        state_space=state_space,
        q_table=learner,
        demands=model.demands,
        simulation_memory=local_memory,
        exploration_method=lambda: next(alws),#target.furthest_from_the_target,
        epsilon=1
    )

    model.router = router
    local_memory.railroad = model
    sim.simulate(model=model, time_horizon=timedelta(days=30))
    learner.save()
    output = OutputData(
        operated_volume=router.operated_volume(),
        total_demand=router.total_demand(),
        process_id=os.getpid(),
        episode_number=episode_number,
        q_table_size=len(learner.q_table)
    )
    output_queue.put(output)
    ...

# @profile
def run_training_loop(output_queue):
    for episode in range(EPISODES_BY_PROCESS):
        run_episode(
            episode_number=episode,
            output_queue=output_queue
        )
    info(f"Trainamento do processo {os.getpid()} finalizado!")

def training_process_wrapper(output_queue):
    p = Process(target=run_training_loop, args=(output_queue,))
    return p


if __name__ == '__main__':
    with Manager() as manager:
        output_queue = manager.Queue()
        experience_queue = manager.Queue()
        # _, _ = setup_shared_components(experience_queue)
        stop_signal_log = Event()

        # iniciar processo
        logging_process = Process(target=logging_loop, args=(stop_signal_log, output_queue))
        logging_process.start()
        logging_pid = logging_process.pid
        stop_signal = Event()

        # learner_process = Process(target=learning_loop, args=(experience_queue, stop_signal))
        # learner_process.start()

        for step in range(TRAINING_STEPS):


            # Inicia os processos dos atores
            actor_processes = [
                training_process_wrapper(output_queue)
                for _ in range(NUM_PROCESSES)
            ]

            for proc in actor_processes:
                proc.start()

            # Aguarda os atores terminarem
            for proc in actor_processes:
                proc.join()
            # sleep(1)
            stop_signal.set()
            # learner_process.join()
            stop_signal.clear()

        # while not output_queue.empty():
        #     continue
        stop_signal_log.set()
        logging_process.join()

