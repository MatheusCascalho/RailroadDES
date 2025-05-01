from models.node_constraints import LiberationConstraintSystem
from models.observers import AbstractObserver
from datetime import timedelta
from models.clock import Clock

class ManeuveringConstraintFactory:
    def __init__(
            self,
            post_operation_time: int,
            clock: Clock
    ):
        if not isinstance(post_operation_time, int):
            raise Exception(f"Post operation should be int, not {type(post_operation_time)}")
        self.post_operation_time = timedelta(hours=post_operation_time)
        self.clock = clock

    def create(self, train_id):
        constraint = PostOperativeManeuverConstraint(
            post_operation_time=self.post_operation_time,
            clock=self.clock,
            train_id=train_id
        )
        return constraint

class PostOperativeManeuverConstraint(LiberationConstraintSystem):
    def __init__(
            self,
            post_operation_time: timedelta,
            clock: Clock,
            train_id: str
    ):
        self.train_id = train_id
        self.post_operation_time = post_operation_time
        self.clock = clock
        self.start = self.clock.current_time
        super().__init__()

    def update(self):
        elapsed_time = self.clock.current_time - self.start
        if elapsed_time >= self.post_operation_time:
            self.state_machine.update("free_up")
