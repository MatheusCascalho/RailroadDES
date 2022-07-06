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
        self.transitions = np.array(transitions)
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
        mark = [p.tokens for p in self.places]
        return np.array(mark)

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
        cardinality = (
            len(self.transitions),
            len(self.places)
        )
        matrix = np.zeros(cardinality)
        for arc in self.arcs:
            if isinstance(arc.output, Transition):
                i = self.indexes['transitions'][arc.output.identifier]
                j = self.indexes['places'][arc.input.identifier]
                matrix[i, j] = - arc.weight
        return matrix

    def a_plus(self):
        ...

    def update(self, transition):
        transition_index = self.indexes['transitions'][transition.identifier]
        incidence_matrix = self.incidence_matrix()
        current_marking = self.marking
        new_marking = current_marking + incidence_matrix[transition_index]
        for i, new_tokens in enumerate(new_marking):
            self.places[i].update(tokens=new_tokens)
        return self.marking

    def allowed_transitions(self):
        current_marking = self.marking
        a_minus = self.a_minus()
        places_amount = len(self.places)
        reference_vector = np.zeros(places_amount)

        debit_tokens = current_marking + a_minus
        valid_transitions = np.where(
            np.all(debit_tokens >= reference_vector, axis=1)
        )
        allowed = self.transitions[valid_transitions]
        return allowed


    def coverage_tree(self):
        ...

    def composition(self, net):
        ...

