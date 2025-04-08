from dataclasses import dataclass, field
from typing import Any, Callable, Union
from abc import ABCMeta, abstractmethod

class AbstractObserver:
    @abstractmethod
    def update(self, *args):
        pass



@dataclass
class State:
    name: Any
    is_current_state: bool
    observers: list[AbstractObserver] = field(default=list)


    def __bool__(self):
        return self.is_current_state

    def activate(self):
        self.is_current_state = True
        self.notify()

    def deactivate(self):
        self.is_current_state = False
        self.notify()

    def notify(self):
        for observer in self.observers:
            observer.update(self)


@dataclass
class Transition(AbstractObserver):
    name: str
    origin: State
    trigger: bool = field(init=False)
    action: Callable
    destination: State

    def update(self, state: State):
        self.trigger = state.is_current_state
        if self.trigger:
            self.origin.deactivate()
            self.destination.activate()

    def force_trigger(self):
        self.trigger = True
        if self.trigger:
            self.origin.deactivate()
            self.destination.activate()


class StateMachine:
    def __init__(
            self,
            transitions: list[Transition],
    ):
        ...

    def allowed_transitions(self) -> list[Transition]:
        ...

    def current_state(self) -> Any:
        ...

    def update(self):
        ...

