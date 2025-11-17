from src.standards.state_machine import (
    StateMachine,
    Transition,
    State
)
from src.standards.observers import AbstractObserver
import pytest
import random


def test_state_machine_should_should_update_state_when_event_trigger_happen_and_transition_is_allowed():
    on = State("ON", False)
    off = State("OFF", True)

    t1 = Transition(
        name="Turn ON",
        origin=off,
        destination=on,
        action=lambda:print('ligando!')
    )
    t2 = Transition(
        name="Turn OFF",
        origin=on,
        destination=off,
        action=lambda : print('desligando!')
    )
    transitions = [t1, t2]

    allow_turn_on = State("Turn ON", False)
    allow_turn_on.add_observers([t1])
    state_machine = StateMachine(
        transitions=transitions,
    )

    # Act
    allow_turn_on.activate()

    # Assert
    assert state_machine.current_state.name == "ON"


##### TESTES PROPOSTOS PELO CHAT GPT
##### TEST STATE ####################

def test_state_creation():
    state = State(name="Estado A", is_marked=False)
    assert state.name == "Estado A"
    assert not state.is_marked

def test_activate_state():
    state = State(name="Estado A", is_marked=False)
    state.activate()
    assert state.is_marked

def test_deactivate_state():
    state = State(name="Estado A", is_marked=True)
    state.deactivate()
    assert not state.is_marked

def test_notify_on_state_change():
    class TestObserver(AbstractObserver):
        def __init__(self):
            self.subjects = []
        def update(self, *args):
            self.called = True

    state = State(name="Estado A", is_marked=False)
    observer = TestObserver()
    state.add_observers([observer])

    state.activate()
    assert observer.called


##### TEST Transition ####################
def test_transition_creation():
    state1 = State(name="Estado A", is_marked=False)
    state2 = State(name="Estado B", is_marked=False)
    transition = Transition(name="A->B", origin=state1, action=lambda: None, destination=state2)

    assert transition.name == "A->B"
    assert transition.origin == state1
    assert transition.destination == state2


def test_update_trigger():
    # Nesse teste a transição é disparada assim que estado fica marcado
    state1 = State(name="Estado A", is_marked=True)
    state2 = State(name="Estado B", is_marked=False)
    transition = Transition(name="A->B", origin=state1, action=lambda: None, destination=state2)

    state1.add_observers([transition])
    transition.update()

    assert state2.is_marked
    assert not state1.is_marked


def test_force_trigger():
    state1 = State(name="Estado A", is_marked=True)
    state2 = State(name="Estado B", is_marked=False)
    transition = Transition(name="A->B", origin=state1, action=lambda: None, destination=state2)

    transition.force_trigger()

    assert state2.is_marked
    assert not state1.is_marked

##### TEST StateMachine ####################
def test_initial_state():
    state1 = State(name="Estado A", is_marked=True)
    state2 = State(name="Estado B", is_marked=False)
    transition = Transition(name="A->B", origin=state1, action=lambda: None, destination=state2)
    sm = StateMachine(transitions=[transition])

    assert sm.current_state == state1

def test_allowed_transitions():
    state1 = State(name="Estado A", is_marked=True)
    state2 = State(name="Estado B", is_marked=False)
    transition = Transition(name="A->B", origin=state1, action=lambda: None, destination=state2)
    transition2 = Transition(name="B->A", origin=state2, action=lambda: None, destination=state1)
    sm = StateMachine(transitions=[transition, transition2])

    transitions = sm.allowed_transitions()
    assert len(transitions) == 1
    assert transitions[0] == transition

def test_state_transition():
    state1 = State(name="Estado A", is_marked=True)
    state2 = State(name="Estado B", is_marked=False)
    transition = Transition(name="A->B", origin=state1, action=lambda: None, destination=state2)
    state1.add_observers([transition])
    sm = StateMachine(transitions=[transition])

    sm.current_state = state1
    sm.allowed_transitions()[0].force_trigger()

    assert sm.current_state == state2


### TESTES RANDOMIZADOS

@pytest.fixture
def random_state():
    """Retorna um estado aleatório."""
    return State(name=f"Estado {random.randint(1, 100)}", is_marked=random.choice([True, False]))

@pytest.fixture
def random_state_factory():
    def make():
        return State(name=f"Estado {random.randint(1, 100)}", is_marked=random.choice([True, False]))
    return make

@pytest.fixture
def random_transition_factory(random_state_factory):
    """Cria uma transição aleatória entre dois estados diferentes."""
    states = [random_state_factory() for _ in range(10)]
    def make():
        origin = random.choice(states)
        destination = random.choice(states)
        while origin == destination:
            destination = random.choice(states)
        return Transition(
            name=f"Transição de {origin.name} para {destination.name}",
            origin=origin,
            action=lambda: None,
            destination=destination
        )
    return make

@pytest.fixture
def random_transition(random_state_factory):
    """Cria uma transição aleatória entre dois estados diferentes."""
    origin = random_state_factory()
    destination = random_state_factory()
    while origin == destination:
        destination = random_state_factory()
    return Transition(
        name=f"Transição de {origin.name} para {destination.name}",
        origin=origin,
        action=lambda: None,
        destination=destination
    )

def test_random_state_creation(random_state):
    assert random_state.name.startswith("Estado")
    assert random_state.is_marked in [True, False]

def test_random_transition_creation(random_transition):
    assert random_transition.origin != random_transition.destination

# @pytest.mark.parametrize("num_states", [5, 10, 15])  # Testar com diferentes números de estados
# def test_random_state_machine(num_states, random_state_factory, random_transition_factory):
#     # Criar transições aleatórias entre os estados
#     transitions = [random_transition_factory() for _ in range(num_states)]
#     # Criar um número aleatório de estados
#     states = list(set([t.origin for t in transitions] + [t.destination for t in transitions]))
#
#     # Criar a máquina de estados
#     sm = StateMachine(transitions=transitions)
#
#     # Testar o comportamento de transições aleatórias
#     for _ in range(100):  # Fazer 100 transições aleatórias
#         state = random.choice(states)
#         if state.is_marked:
#             sm.current_state = state
#             allowed = sm.allowed_transitions()
#             if allowed:
#                 transition = random.choice(allowed)
#                 transition.force_trigger()
#
#     # Verificar se a máquina de estados sempre tem um estado válido ativo
#     current_state = sm.current_state
#     assert current_state is not None
#     assert current_state.is_marked