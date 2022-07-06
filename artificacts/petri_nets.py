from petri_nets.petri_components import Place, Transition, Arc
from petri_nets.net import PetriNet
import numpy as np
import pytest


@pytest.fixture
def simple_net():
    p1 = Place(
        tokens=0,
        meaning="",
        identifier="p1"
    )
    p2 = Place(
        tokens=0,
        meaning="",
        identifier="p2"
    )
    p3 = Place(
        tokens=1,
        meaning="",
        identifier="p3"
    )

    t1 = Transition(
        identifier="t1",
        intrinsic_time=None,
        input_places=[],
        output_places=[],
        meaning="",
    )
    t2 = Transition(
        identifier="t2",
        intrinsic_time=None,
        input_places=[],
        output_places=[],
        meaning="",
    )
    t3 = Transition(
        identifier="t3",
        intrinsic_time=None,
        input_places=[],
        output_places=[],
        meaning="",
    )

    arcs = [
        Arc(input=t1, output=p1),
        Arc(input=p1, output=t2),
        Arc(input=t2, output=p2),
        Arc(input=p2, output=t3),
        Arc(input=t3, output=p3),
        Arc(input=p3, output=t2),
    ]
    places = [
        p1, p2, p3
    ]
    transitions = [
        t1, t2, t3
    ]

    net = PetriNet(
        places=places,
        transitions=transitions,
        arcs=arcs
    )

    return net