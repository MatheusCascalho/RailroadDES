from dataclasses import dataclass, field
from typing import Any, Callable, Union
from abc import ABCMeta, abstractmethod

class AbstractObserver:
    subjects: list

    @abstractmethod
    def update(self, *args):
        pass


@dataclass
class State:
    name: Any
    is_marked: bool
    observers: list[AbstractObserver] = field(default_factory=list)


    def __bool__(self):
        return self.is_marked

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, State):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return False

    def activate(self):
        if self.is_marked:
            raise Exception("Estado já está ativo!")
        self.is_marked = True
        self.notify()

    def deactivate(self):
        if not self.is_marked:
            raise Exception("Estado não está ativo!")
        self.is_marked = False
        self.notify()

    def notify(self):
        for observer in self.observers:
            observer.update(self)

    def add_observer(self, observer: list[AbstractObserver]):
        self.observers.extend(observer)
        for observer in observer:
            if self not in observer.subjects:
                observer.subjects.append(self)


@dataclass
class Transition(AbstractObserver):
    name: str
    origin: State
    trigger: bool = field(init=False, default_factory=bool)
    action: Callable
    destination: State
    subjects: list = field(init=False, default_factory=list)

    def update(self, state: State):
        self.trigger = state.is_marked
        if self.trigger and self.origin.is_marked:
            self.origin.deactivate()
            self.destination.activate()
            self.trigger = False

    def force_trigger(self):
        self.trigger = True
        if self.trigger and self.origin.is_marked:
            self.origin.deactivate()
            self.destination.activate()
            self.trigger = False


class StateMachine:
    def __init__(
            self,
            transitions: list[Transition],
    ):
        self.machine = {}
        self.transitions = {
            t.name: t
            for t in transitions
        }
        for transition in transitions:
            if transition.origin not in self.machine:
                self.machine[transition.origin] = [transition]
            else:
                self.machine[transition.origin].append(transition)
            if transition.destination not in self.machine:
                self.machine[transition.destination] = []

        self.__current_state = self.get_current_state()

    @property
    def current_state(self):
        if self.__current_state.is_marked:
            return self.__current_state
        else:
            self.__current_state = self.get_current_state()
            return self.__current_state

    @current_state.setter
    def current_state(self, value: State):
        if not value.is_marked:
            raise Exception("O estado deve estar marcado para ser atribuido como estado atual do sistema")
        self.__current_state = value

    def allowed_transitions(self) -> list[Transition]:
        if not self.current_state.is_marked:
            self.current_state = self.get_current_state()
        return self.machine[self.current_state]

    def get_current_state(self) -> State:
        for state in self.machine:
            if state.is_marked:
                return state
        raise Exception('Nenhum estado está ativado')


