from models.discrete_event_system import DiscreteEventSystem
from models.state_machine import State, StateMachine, MultiCriteriaTransition
from models.states import NodeProcessState
from abc import abstractmethod
from models.constants import Process

class ProcessConstraintSystem(DiscreteEventSystem):
    def __init__(
            self,
    ):
        super().__init__()


    def build_state_machine(self) -> StateMachine:
        ready = State(name=NodeProcessState.READY, is_marked=True)
        busy = State(name=NodeProcessState.BUSY, is_marked=False)
        blocked = State(name=NodeProcessState.BLOCKED, is_marked=False)

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

    def is_blocked(self):
        return self.state_machine.current_state.name == NodeProcessState.BLOCKED

    @abstractmethod
    def process_type(self) -> Process:
        pass


