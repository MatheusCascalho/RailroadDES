from src.target import TargetManager, SimpleTargetManager
from src.task import Task
from src.demand import Demand
from src.router import Router


class TargetRouter(Router):
    def __init__(
            self,
            demands: list[Demand],
            target_manager_factory: callable = SimpleTargetManager
    ):
        super().__init__(demands=demands)
        self.target_manager: TargetManager = target_manager_factory(demands)


    def choose_task(self, current_time, train_size, model_state):
        furthest_demand = self.target_manager.furthest_from_the_target()
        task = Task(
            demand=furthest_demand,
            path=[furthest_demand.flow.origin, furthest_demand.flow.destination],
            task_volume=train_size,
            current_time=current_time,
            state=model_state
        )
        return task