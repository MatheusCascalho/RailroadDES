from abc import abstractmethod
from datetime import timedelta
from random import randint
import stat
from typing import Callable, Generator
import torch

from dataclasses import dataclass
from copy import copy
from src.DQNRouter import Learner
from src.demand import Flow
from src.solution_based_router import Solution, SolutionBasedRouter
import numpy as np
import logging
import os 
import time

class Neighborhood:
    def __init__(self, solution: Solution) -> None:
        self.solution = solution

    @abstractmethod
    def neighborhood(self) -> Generator[Solution, None, None]:
        pass


class SwapToFast(Neighborhood):
    def __init__(self, solution: Solution, router: SolutionBasedRouter, learner: Learner, k) -> None:
        super().__init__(solution)
        self.router = router
        self.learner = learner
        self.k = k

    def neighborhood(self):
        solution_size = len(self.solution.flow_sequence)
        penalties = [t.penalty() for t in self.router.choosed_tasks[:solution_size]]
        worst_indexes = np.argsort(-np.array(penalties))
        for worst_index in worst_indexes:
            if worst_index == len(penalties)-1:
                continue
            if penalties[worst_index] == timedelta():
                break
            if self.k == 'DQN':
                state = self.router.choosed_tasks[worst_index].model_state
            else:
                state = None
            if state is not None:
                flow = self.choose_flow(state=state, current_flow=self.solution.flow_sequence[worst_index])
                new_solution = copy(self.solution.flow_sequence)
                new_solution.append(new_solution[worst_index])
                new_solution[worst_index] = flow #self.solution.flow_sequence[best_index]
                # new_solution[best_index] = self.solution.flow_sequence[worst_index]
                new_solution = Solution(flow_sequence=new_solution)
                yield new_solution
            else:
                cycle_times =np.array([
                    self.router.get_cycle_time(f).total_seconds()
                    for f in self.solution.flow_sequence[worst_index+1:]
                ])
                same_flow = np.array(self.solution.flow_sequence[worst_index+1:])==self.solution.flow_sequence[worst_index]
                cycle_times[same_flow] = np.inf
                best_index = worst_index + np.argmin(cycle_times) + 1
                new_solution = copy(self.solution.flow_sequence)
                new_solution[worst_index] = self.solution.flow_sequence[best_index]
                new_solution[best_index] = self.solution.flow_sequence[worst_index]
                new_solution = Solution(flow_sequence=new_solution)
                yield new_solution
            penalties[worst_index] = timedelta()

    def choose_flow(self, state, current_flow):
        state = self.learner.state_space.to_array(state)
        state = torch.FloatTensor(state).unsqueeze(0)
        # demand_index = self.policy_net(state).argmax().item()
        # selected_demand = self.action_space.get_demand(demand_index)
        # Passa o estado pela rede
        q_values = self.learner.policy_net(state).detach().cpu().numpy().flatten()

        # Ordena os índices das ações pelo valor Q (do maior para o menor)
        sorted_indices = q_values.argsort()[::-1]

        # Se quiser, também pode obter os Q-values ordenados:
        sorted_q_values = q_values[sorted_indices]

        # Exemplo: selecionar o melhor, o segundo melhor, etc.
        i = 0
        while i < len(self.learner.action_space.demands):
            best_action_index = sorted_indices[i]
            try:
                # Exemplo: obter a ação correspondente ao melhor Q
                selected_demand = self.learner.action_space.get_demand(best_action_index)
                if selected_demand.flow == current_flow:
                    i += 1
                    continue
                return selected_demand.flow
            except:
                i += 1        



class Move(Neighborhood):
    def neighborhood(self):
        while True:
            idx = randint(0, len(self.solution.flow_sequence)-1)
            new_idx = randint(0, len(self.solution.flow_sequence)-1)
            if new_idx == idx:
                new_idx = randint(0, len(self.solution.flow_sequence)-1)
            new_solution = self.solution.flow_sequence
            v = new_solution.pop(idx)
            new_solution.insert(new_idx, v)
            new_solution = Solution(new_solution)
            yield new_solution


@dataclass
class solution_register:
    solution: Solution
    fitness: float
            
class VNSalgorithm:
    def __init__(
            self, 
            solution: Solution, 
            neighborhoods: list[Callable[[Solution], Neighborhood]],
            objective_function: Callable[[Solution], float]
        ) -> None:
        self.neighborhoods_structures = neighborhoods
        self.neighborhoods = [n(solution).neighborhood() for n in neighborhoods]
        self.__solution = solution
        self.objective_function = objective_function
        start = time.time()
        self.current_fitness = objective_function(solution)
        end = time.time()
        elapsed_time = end - start
        
        # Configuração de logger
        self.logger = logging.getLogger('VNS')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            log_dir = "logs/vns"
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(log_dir, f"vns.log"))
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            sh = logging.StreamHandler()
            self.logger.addHandler(sh)
        self.logger.info(
            f"[step=-1] "
            f"k=-1 | "
            f"neighborhood=SOLUCAO INICIAL | "
            f"fitness_logistic_time={self.current_fitness:.2f} | "
            f"is_local_optimal=False | "
            f"start={start:.4f} | "
            f"end={end:.4f} | "
            f"simulation_time={elapsed_time:.4f} | "
        )

    @property
    def solution(self):
        return self.__solution
    
    @solution.setter
    def solution(self, solution):
        self.neighborhoods = [n(solution).neighborhood() for n in self.neighborhoods_structures]
        self.__solution = solution

    def best_improvement(self, k, it):
        n = 0
        best_solution = self.solution
        best_fitness = self.current_fitness
        for neighbor in self.neighborhoods[k]:
            start = time.time()
            fitness = self.objective_function(neighbor)
            end = time.time()
            elapsed_time = end - start
            self.logger.info(
                f"[step={it}] "
                f"k={k} | "
                f"neighbor={n} | "
                f"neighborhood={self.neighborhoods_structures[k].__qualname__} | "
                f"fitness_logistic_time={fitness:.2f} | "
                f"is_local_optimal={fitness < best_fitness} | "
                f"start={start:.4f} | "
                f"end={end:.4f} | "
                f"simulation_time={elapsed_time:.4f} | "
            )
            n+=1
            if fitness < best_fitness:
                best_solution= neighbor
                best_fitness = fitness
        self.solution = best_solution
        self.current_fitness = best_fitness
        return neighbor, fitness

    def first_improvement(self, k, it):
        n = 0
        for neighbor in self.neighborhoods[k]:
            start = time.time()
            fitness = self.objective_function(neighbor)
            end = time.time()
            elapsed_time = end - start
            self.logger.info(
                f"[step={it}] "
                f"k={k} | "
                f"neighbor={n} | "
                f"neighborhood={self.neighborhoods_structures[k].__qualname__} | "
                f"fitness_logistic_time={fitness:.2f} | "
                f"is_local_optimal={fitness < self.current_fitness} | "
                f"start={start:.4f} | "
                f"end={end:.4f} | "
                f"simulation_time={elapsed_time:.4f} | "
            )
            n+=1
            if fitness < self.current_fitness:
                self.solution = neighbor
                self.current_fitness = fitness
                self.logger.info(f'Current Best Solution: {self.solution}')
                return neighbor, fitness
        return self.solution, self.current_fitness

    def solve(self, max_iterations=20):
        it = 0
        
        while it < max_iterations:
            k = 0
            while k < len(self.neighborhoods):
                # new_solution = local_search(new_solution)
                # self.current_fitness = self.objective_function(self.solution)
                try:
                    new_solution, new_fitness = self.first_improvement(k=k, it=it)
                    # new_solution, new_fitness = self.best_improvement(k=k, it=it)
                    it += 1
                    if new_fitness < self.current_fitness:                        
                        if k>0:
                            k = 0  # volta à primeira vizinhança
                    else:
                        k += 1
                    if self.current_fitness <= 31:
                        break
                except StopIteration:
                    self.neighborhoods.append(Move(self.solution).neighborhood)
                    self.neighborhoods_structures.append(Move)
                    k += 1
                

            