from models.state_machine import (
    StateMachine,
    Transition,
    State
)


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
    allow_turn_on.add_observer([t1])
    state_machine = StateMachine(
        transitions=transitions,
    )

    # Act
    allow_turn_on.activate()

    # Assert
    assert state_machine.current_state.name == "ON"