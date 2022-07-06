from petri_nets.petri_components import Place, Transition, Arc
from typing import List
import numpy as np

class PetriNet:
    def __init__(
            self,
            places: List[Place],
            transitions: List[Transition],
            arcs: List[Arc]
    ):
        self.transitions = transitions
        self.arcs = arcs
        self.places = places
        self.indexes = {
            "places": {
                place.identifier: i for i, place in enumerate(places)
            },
            "transitions": {
                transition.identifier: i for i, transition in enumerate(transitions)
            },
        }

    @property
    def marking(self):
        ...

    def incidence_matrix(self):
        cardinality = (
            len(self.transitions),
            len(self.places)
        )
        matrix = np.zeros(cardinality)
        for arc in self.arcs:
            if isinstance(arc.input, Transition):
                i = self.indexes['transitions'][arc.input.identifier]
                j = self.indexes['places'][arc.output.identifier]
                matrix[i, j] = arc.weight
            else:
                i = self.indexes['transitions'][arc.output.identifier]
                j = self.indexes['places'][arc.input.identifier]
                matrix[i, j] = - arc.weight
        return matrix

    def a_minus(self):
        ...

    def a_plus(self):
        ...




    def update(self):
        ...

    def allowed_transitions(self):
        ...

    def coverage_tree(self):
        ...

    def composition(self, net):
        ...

