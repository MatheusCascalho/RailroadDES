from typing import Union

from interfaces.train_interface import TrainInterface
from models import model_queue as mq
from models.clock import Clock
from models.constants import Process
from models.state_machine import State, Transition, StateMachine, MultiCriteriaTransition
from models.states import ProcessorState, NodeProcesState, LoadState, ActivityState
from abc import abstractmethod


class DiscreteEventSystem:
    def __init__(self, **kwargs):
        self.state_machine = self.build_state_machine(**kwargs)

    @abstractmethod
    def build_state_machine(self, **kwargs):
        pass

    def __str__(self):
        return str(self.state_machine.current_state)

    __repr__ = __str__


class ProcessorSystem(DiscreteEventSystem):
    def __init__(self, processor_type: Process, queue_to_leave: mq.Queue, clock: Clock):
        self.type = processor_type
        self.current_train: Union[None, TrainInterface] = None
        self.queue_to_leave = queue_to_leave
        self.clock = clock
        super().__init__()


    def build_state_machine(self):
        idle = State(name=ProcessorState.IDLE, is_marked=True)
        busy = State(name=ProcessorState.BUSY, is_marked=False)
        put_in = Transition(name="Put in", origin=idle, destination=busy)
        free_up = Transition(name="Free Up", origin=busy, destination=idle, action=self.clear)
        sm = StateMachine(transitions=[put_in, free_up])
        return sm

    def put_in(self, train: TrainInterface):
        if self.state_machine.current_state.name == ProcessorState.IDLE:
            self.current_train = train
            self.state_machine.update()
        else:
            raise Exception("Processor is Busy")

    def free_up(self):

        if self.state_machine.current_state.name == ProcessorState.BUSY:
            self.current_train = None
            self.state_machine.update()
        else:
            raise Exception("Processor is Idle")

    def clear(self):
        train = self.current_train
        self.queue_to_leave.push(
            train,
            arrive=self.clock.current_time
        )
        return train

    def is_ready_to_clear(self):
        if self.current_train:
            return self.current_train.process_end <= self.clock.current_time
        return False

    @property
    def is_idle(self):
        return self.state_machine.current_state.name == ProcessorState.IDLE


class ProcessConstraintSystem(DiscreteEventSystem):
    def __init__(
            self,
    ):
        super().__init__()

    def build_state_machine(self) -> StateMachine:
        ready = State(name=NodeProcesState.READY, is_marked=True)
        busy = State(name=NodeProcesState.BUSY, is_marked=False)
        blocked = State(name=NodeProcesState.BLOCKED, is_marked=False)

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
        return self.state_machine.current_state.name == NodeProcesState.BLOCKED


class LoadSystem(DiscreteEventSystem):
    def __init__(
            self,
            capacity: float,
            is_loaded: bool,
            processing_state: State,
            queue_to_leave_state: State
    ):
        self.capacity = capacity
        self.volume = capacity if is_loaded else 0
        super().__init__(
            processing_state=processing_state,
            queue_to_leave_state=queue_to_leave_state,
            is_loaded=is_loaded
        )

    def build_state_machine(self, processing_state, queue_to_leave_state, is_loaded):
        loaded = State(name=LoadState.LOADED, is_marked=is_loaded)
        loading = State(name=LoadState.LOADING, is_marked=False)
        empty = State(name=LoadState.EMPTY, is_marked=not is_loaded)
        unloading = State(name=LoadState.UNLOADING, is_marked=False)
        to_loaded = Transition(
            name="to_loaded",
            origin=loading,
            destination=loaded,
            action=self.load
        )
        to_unloading = Transition(
            name="to_unloading",
            origin=loaded,
            destination=unloading
        )
        to_empty = Transition(
            name="to_empty",
            origin=unloading,
            destination=empty,
            action=self.unload
        )
        to_loading = Transition(
            name="to_loading",
            origin=empty,
            destination=loading
        )

        processing_state.add_observer([to_loading, to_unloading])
        queue_to_leave_state.add_observer([to_loaded, to_empty])
        transitions = [to_loaded, to_empty, to_loading, to_unloading]
        sm = StateMachine(transitions=transitions)
        return sm

    def set_processing_state(self, processing_state: State):
        loading = self.state_machine.transitions['to_loading']
        unloading = self.state_machine.transitions['to_unloading']
        processing_state.add_observer([loading, unloading])

    def set_in_queue_state(self, in_queue_state: State):
        loaded = self.state_machine.transitions['to_loaded']
        empty = self.state_machine.transitions['to_empty']
        in_queue_state.add_observer([loaded, empty])

    def load(self):
        self.volume = self.capacity

    def unload(self):
        self.volume = 0


class ActivitySystem(DiscreteEventSystem):
    def __init__(self, path):
        self.__path = path
        self.current_location = path[0]
        super().__init__()

    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, new_path):
        self.__path = new_path
        self.current_location = self.__path[0]

    def build_state_machine(self):
        moving = State(name=ActivityState.MOVING, is_marked=False)
        queue_to_enter = State(name=ActivityState.QUEUE_TO_ENTER, is_marked=True)
        processing = State(name=ActivityState.PROCESSING, is_marked=False)
        queue_to_leave = State(name=ActivityState.QUEUE_TO_LEAVE, is_marked=False)

        leave = Transition(
            name="leave",
            origin=queue_to_leave,
            destination=moving
        )
        arrive = Transition(
            name="arrive",
            origin=moving,
            destination=queue_to_enter
        )
        start = Transition(
            name="start",
            origin=queue_to_enter,
            destination=processing
        )
        finish = Transition(
            name="finish",
            origin=processing,
            destination=queue_to_leave
        )
        transitions = [
            leave, arrive,
            start, finish
        ]
        sm = StateMachine(transitions=transitions)
        return sm

    @property
    def processing_state(self):
        return self.state_machine.states[ActivityState.PROCESSING]


    @property
    def queue_to_leave_state(self):
        return self.state_machine.states[ActivityState.QUEUE_TO_LEAVE]


    def finish_process(self):
        if self.state_machine.current_state.name == ActivityState.PROCESSING:
            self.state_machine.update()

    def start_process(self):
        if self.state_machine.current_state.name == ActivityState.QUEUE_TO_ENTER:
            self.state_machine.update()

    def leave(self):
        if self.state_machine.current_state.name == ActivityState.QUEUE_TO_LEAVE:
            self.state_machine.update()

    def arrive(self):
        if self.state_machine.current_state.name == ActivityState.MOVING:
            self.state_machine.update()
            self.path = self.path[1:]
