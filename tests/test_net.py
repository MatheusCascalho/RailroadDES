from petri_nets.petri_components import Place, Transition, Arc
from petri_nets.net import PetriNet
import numpy as np
from numpy import testing


def test_node_net():
    process_unit = Place(
        tokens=0,
        meaning="process unit",
        identifier="LOAD_POINT_1"
    )
    receiving_vacancy = Place(
        tokens=0,
        meaning="receiving_vacancy",
        identifier="RECEIVE_YARD_VACCANCY_POINT_1"
    )
    receiving_occupancy = Place(
        tokens=0,
        meaning="receiving_occupancy",
        identifier="RECEIVE_YARD_POINT_1"
    )
    dispatch_vacancy = Place(
        tokens=0,
        meaning="dispatch_vacancy",
        identifier="DISPATCH_YARD_VACCANCY_POINT_1"
    )
    dispatch_occupancy = Place(
        tokens=0,
        meaning="dispatch_occupancy",
        identifier="DISPATCH_YARD_POINT_1"
    )

    call_to_enter = Transition(
        identifier="enter_to_load_1",
        intrinsic_time=None,
        input_places=[],
        output_places=[],
        meaning="",
    )
    process = Transition(
        identifier="process_in_load_1",
        intrinsic_time=None,
        input_places=[],
        output_places=[],
        meaning="",
    )
    release = Transition(
        identifier="release_from_load_1",
        intrinsic_time=None,
        input_places=[],
        output_places=[],
        meaning="",
    )

    arcs = [
        Arc(input=receiving_occupancy, output=call_to_enter),
        Arc(input=call_to_enter, output=process_unit),
        Arc(input=process_unit, output=process),
        Arc(input=process, output=dispatch_occupancy),
        Arc(input=dispatch_occupancy, output=release),
        Arc(input=release, output=dispatch_vacancy),
        Arc(input=call_to_enter, output=receiving_vacancy),
    ]
    places = [
        receiving_vacancy,
        receiving_occupancy,
        process_unit,
        dispatch_occupancy,
        dispatch_vacancy
    ]
    transitions = [
        call_to_enter,
        process,
        release
    ]

    net = PetriNet(
        places=places,
        transitions=transitions,
        arcs=arcs
    )
    assert False


def test_simple_net():
    p1 = Place(
        tokens=0,
        meaning="process unit",
        identifier="LOAD_POINT_1"
    )
    p2 = Place(
        tokens=0,
        meaning="receiving_vacancy",
        identifier="RECEIVE_YARD_VACCANCY_POINT_1"
    )
    p3 = Place(
        tokens=1,
        meaning="receiving_occupancy",
        identifier="RECEIVE_YARD_POINT_1"
    )

    t1 = Transition(
        identifier="enter_to_load_1",
        intrinsic_time=None,
        input_places=[],
        output_places=[],
        meaning="",
    )
    t2 = Transition(
        identifier="process_in_load_1",
        intrinsic_time=None,
        input_places=[],
        output_places=[],
        meaning="",
    )
    t3 = Transition(
        identifier="release_from_load_1",
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

    actual = net.incidence_matrix()
    expected = np.array([
        [1, 0, 0],
        [-1, 1, -1],
        [0, -1, 1]
    ])
    testing.assert_equal(actual, expected)


def test_simple_net_marking():
    p1 = Place(
        tokens=0,
        meaning="process unit",
        identifier="LOAD_POINT_1"
    )
    p2 = Place(
        tokens=0,
        meaning="receiving_vacancy",
        identifier="RECEIVE_YARD_VACCANCY_POINT_1"
    )
    p3 = Place(
        tokens=1,
        meaning="receiving_occupancy",
        identifier="RECEIVE_YARD_POINT_1"
    )

    t1 = Transition(
        identifier="enter_to_load_1",
        intrinsic_time=None,
        input_places=[],
        output_places=[],
        meaning="",
    )
    t2 = Transition(
        identifier="process_in_load_1",
        intrinsic_time=None,
        input_places=[],
        output_places=[],
        meaning="",
    )
    t3 = Transition(
        identifier="release_from_load_1",
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

    actual = net.marking
    expected = np.array([0, 0, 1])
    testing.assert_equal(actual, expected)


def test_simple_net_a_minus():
    p1 = Place(
        tokens=0,
        meaning="process unit",
        identifier="LOAD_POINT_1"
    )
    p2 = Place(
        tokens=0,
        meaning="receiving_vacancy",
        identifier="RECEIVE_YARD_VACCANCY_POINT_1"
    )
    p3 = Place(
        tokens=1,
        meaning="receiving_occupancy",
        identifier="RECEIVE_YARD_POINT_1"
    )

    t1 = Transition(
        identifier="enter_to_load_1",
        intrinsic_time=None,
        input_places=[],
        output_places=[],
        meaning="",
    )
    t2 = Transition(
        identifier="process_in_load_1",
        intrinsic_time=None,
        input_places=[],
        output_places=[],
        meaning="",
    )
    t3 = Transition(
        identifier="release_from_load_1",
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

    actual = net.a_minus()
    expected = np.array([
        [0, 0, 0],
        [-1, 0, -1],
        [0, -1, 0]
    ])
    testing.assert_equal(actual, expected)


def test_simple_net_allowed_transitions():
    p1 = Place(
        tokens=1,
        meaning="process unit",
        identifier="LOAD_POINT_1"
    )
    p2 = Place(
        tokens=0,
        meaning="receiving_vacancy",
        identifier="RECEIVE_YARD_VACCANCY_POINT_1"
    )
    p3 = Place(
        tokens=1,
        meaning="receiving_occupancy",
        identifier="RECEIVE_YARD_POINT_1"
    )

    t1 = Transition(
        identifier="enter_to_load_1",
        intrinsic_time=None,
        input_places=[],
        output_places=[],
        meaning="",
    )
    t2 = Transition(
        identifier="process_in_load_1",
        intrinsic_time=None,
        input_places=[],
        output_places=[],
        meaning="",
    )
    t3 = Transition(
        identifier="release_from_load_1",
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

    actual = net.allowed_transitions()
    expected = np.array([t1, t2])

    testing.assert_equal(actual, expected)


def test_simple_net_update_marking():
    p1 = Place(
        tokens=1,
        meaning="process unit",
        identifier="LOAD_POINT_1"
    )
    p2 = Place(
        tokens=0,
        meaning="receiving_vacancy",
        identifier="RECEIVE_YARD_VACCANCY_POINT_1"
    )
    p3 = Place(
        tokens=1,
        meaning="receiving_occupancy",
        identifier="RECEIVE_YARD_POINT_1"
    )

    t1 = Transition(
        identifier="enter_to_load_1",
        intrinsic_time=None,
        input_places=[],
        output_places=[],
        meaning="",
    )
    t2 = Transition(
        identifier="process_in_load_1",
        intrinsic_time=None,
        input_places=[],
        output_places=[],
        meaning="",
    )
    t3 = Transition(
        identifier="release_from_load_1",
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

    actual = net.update(t2)
    expected = np.array([0, 1, 0])

    testing.assert_equal(actual, expected)


def test_coverage_tree():
    p1 = Place(
        tokens=1,
        meaning="",
        identifier="p1"
    )
    p2 = Place(
        tokens=0,
        meaning="",
        identifier="p2"
    )
    p3 = Place(
        tokens=0,
        meaning="",
        identifier="p3"
    )
    p4 = Place(
        tokens=0,
        meaning="",
        identifier="p4"
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
        Arc(input=p1, output=t1),
        Arc(input=t1, output=p2),
        Arc(input=t1, output=p3),
        Arc(input=p3, output=t3),
        Arc(input=p2, output=t3),
        Arc(input=t3, output=p3),
        Arc(input=t3, output=p4),
        Arc(input=p2, output=t2),
        Arc(input=t2, output=p1),
    ]
    places = [
        p1, p2, p3, p4
    ]
    transitions = [
        t1, t2, t3
    ]

    net = PetriNet(
        places=places,
        transitions=transitions,
        arcs=arcs
    )

    tree = net.coverage_tree()
    actual = {
        father: [str(child) for child in children] for father, children in tree.items()
    }
    expected = {
        '1 0 0 0': ['0 1 1 0'],
        '0 1 1 0': ['1 0 W 0', '0 0 1 1'],
        '0 0 1 1': [],
        '1 0 W 0': ['0 1 W 0'],
        '0 1 W 0': ['1 0 W 0', '0 0 W 1'],
        '0 0 W 1': []
    }
    assert actual == expected