import sys
from concurrent.futures.process import ProcessPoolExecutor

# Adicionando o diretório ao sys.path
sys.path.append('../')

from models.DQNRouter import DQNRouter, Learner, ActionSpace
from models.tfr_state_factory import TFRStateSpaceFactory
from models.system_evolution_memory import RailroadEvolutionMemory, ExperienceProducer
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
from multiprocessing import Event, Process, Pool
from models.dill_queue import DillQueue
import os
from time import sleep
from dataclasses import dataclass

warnings.filterwarnings('ignore')
N_EPISODES = 10
EPISODES_BY_PROCESS = 25
NUM_PROCESSES = 4
TRAINING_STEPS = 10_000

@dataclass
class OutputData:
    operated_volume: float
    total_demand: float
    final_epsilon: float
    process_id: int
    episode_number: int

def setup_shared_components():
    with open('../tests/artifacts/model_to_train_15.dill', 'rb') as f:
        model = dill.load(f)
    experience_queue = DillQueue(maxsize=10_000)  # Limite opcional
    output_queue = DillQueue(maxsize=10_000)  # Limite opcional
    state_space = TFRStateSpaceFactory(model)
    learner = Learner(
        state_space=state_space,
        action_space=ActionSpace(model.demands),
        policy_net_path='../serialized_models/policy_net_150x6_TFRState_v1_TargetBased_parallel.dill',
        target_net_path='../serialized_models/target_net_150x6_TFRState_v1_TargetBased_parallel.dill',
    )
    global_memory = ExperienceProducer(queue=experience_queue)
    return learner, global_memory, experience_queue, output_queue

def learning_loop(learner, queue, stop_event):
    with learner:
        while not stop_event.is_set():
            try:
                experience = queue.get(timeout=1)
                learner.update(experience)
            except:
                info('Experience queue is empty')
                sleep(.1)
                continue

def logging_loop(stop_event, output_queue):
    log_number = 0
    while not stop_event.is_set():
        try:
            output = output_queue.get(timeout=1)
            critical(f'Log {log_number} - Episode {output.episode_number} - PID: {output.process_id} - Volume: {output.operated_volume} - Demanda: {output.total_demand} - epsilon: {output.final_epsilon}')
            log_number += 1
        except:
            info('Episode queue is empty')
            sleep(.1)
            continue

def run_episode(experience_producer, learner, episode_number, output_queue: DillQueue):
    with open('../tests/artifacts/model_to_train_15.dill', 'rb') as f:
        model = dill.load(f)
    target = SimpleTargetManager(demand=model.demands)
    state_space = TFRStateSpaceFactory(model)

    local_memory = RailroadEvolutionMemory()
    event_factory = DecoratedEventFactory(
        pre_method=local_memory.save_previous_state,
        pos_method=local_memory.save_consequence
    )
    calendar = EventCalendar(event_factory=event_factory)
    sim = DESSimulator(clock=model.mesh.load_points[0].clock, calendar=calendar)
    local_memory.add_observers([experience_producer])
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
    sim.simulate(model=model, time_horizon=timedelta(days=30))

    output = OutputData(
        operated_volume=router.operated_volume(),
        total_demand=router.total_demand(),
        final_epsilon=router.epsilon,
        process_id=os.getpid(),
        episode_number=episode_number
    )
    output_queue.put(output)

def run_training_loop(experience_producer,learner, output_queue):
    for episode in range(EPISODES_BY_PROCESS):
        run_episode(
            experience_producer=experience_producer,
            learner=learner,
            episode_number=episode,
            output_queue=output_queue
        )
    info(f"Trainamento do processo {os.getpid()} finalizado!")

def training_process_wrapper(experience_producer,learner, output_queue):
    p = Process(target=run_training_loop, args=(experience_producer,learner,output_queue))
    return p


if __name__ == '__main__':
    learner, global_memory, experience_queue, output_queue = setup_shared_components()
    stop_signal_log = Event()

    # iniciar processo
    logging_process = Process(target=logging_loop, args=(stop_signal_log, output_queue))
    logging_process.start()
    logging_pid = logging_process.pid

    for step in range(TRAINING_STEPS):
        stop_signal = Event()

        learner_process = Process(target=learning_loop, args=(learner, experience_queue, stop_signal))
        learner_process.start()

        # Inicia os processos dos atores
        actor_processes = [
            training_process_wrapper(global_memory, learner, output_queue)
            for _ in range(NUM_PROCESSES)
        ]

        for proc in actor_processes:
            proc.start()

        # Aguarda os atores terminarem
        for proc in actor_processes:
            proc.join()

        stop_signal.set()
        learner_process.join()

    while not output_queue.empty():
        continue
    stop_signal_log.set()
    logging_process.join()
    with open('all_experiences.dill', 'wb') as f:
        dill.dump(global_memory, f)
