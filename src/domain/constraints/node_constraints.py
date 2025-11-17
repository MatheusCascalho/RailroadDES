from src.domain.systems.discrete_event_system import DiscreteEventSystem
from src.standards.state_machine import State, StateMachine, MultiCriteriaTransition
from src.domain.states import ConstraintState
from abc import abstractmethod, ABC
from src.domain.constants import Process
from dataclasses import dataclass
from datetime import timedelta


def constraint_id_gen():
    i = 0
    while True:
        yield f"Constraint {i}"
        i += 1
constraint_id = constraint_id_gen()
class ConstraintSystem(ABC, DiscreteEventSystem):
    def __init__(self):
        self.ID = next(constraint_id) + " " + self.__class__.__name__
        super().__init__()

    def is_blocked(self):
        return self.state_machine.current_state.name == ConstraintState.BLOCKED


class ProcessConstraintSystem(ConstraintSystem):
    def __init__(
            self,
    ):
        super().__init__()

    def build_state_machine(self) -> StateMachine:
        ready = State(name=ConstraintState.READY, is_marked=True)
        busy = State(name=ConstraintState.BUSY, is_marked=False)
        blocked = State(name=ConstraintState.BLOCKED, is_marked=False)

        start = MultiCriteriaTransition(
            name="start",
            origin=ready,
            destination=busy
        )
        finish = MultiCriteriaTransition(
            name="finish",
            origin=busy,
            destination=ready
        )
        block = MultiCriteriaTransition(
            name="block",
            origin=ready,
            destination=blocked
        )
        release = MultiCriteriaTransition(
            name="release",
            origin=blocked,
            destination=ready
        )
        sm = StateMachine(transitions=[
            start,finish,
            block,release
        ])
        return sm

    @abstractmethod
    def process_type(self) -> Process:
        pass

    @abstractmethod
    def reason(self, *args, **kwargs):
        pass

class LiberationConstraintSystem(ConstraintSystem):
    def __init__(
            self,
    ):
        super().__init__()


    def build_state_machine(self) -> StateMachine:
        free = State(name=ConstraintState.FREE, is_marked=False)
        blocked = State(name=ConstraintState.BLOCKED, is_marked=True)

        free_up = MultiCriteriaTransition(
            name="free_up",
            origin=blocked,
            destination=free
        )
        block = MultiCriteriaTransition(
            name="block",
            origin=free,
            destination=blocked
        )

        sm = StateMachine(transitions=[
            block,free_up
        ])
        return sm


@dataclass
class BlockReason:
    constraint: str
    constraint_type: str
    reason: str
    time_to_try_again: timedelta