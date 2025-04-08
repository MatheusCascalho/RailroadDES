from models.state_machine import (
    StateMachine,
    Transition,
    State
)


def test_state_machine_should_should_update_state_for_allowed_transition():
    on = State("ON", False)
    off = State("OFF", True)
    allow_turn_on = State("", True)

    transitions = [
        Transition(
            name="turn on",
            origin=off,
            destination=on,
            action=lambda:print('ligando!')
        ),
        Transition(
            name="turn off",
            origin=on,
            destination=off,
            action=lambda : print('desligando!')
        )
    ]
    state_machine = StateMachine(
        transitions=transitions,
    )

    # Act
    state_machine.update()
    assert False