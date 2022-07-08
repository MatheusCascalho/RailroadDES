from dataclasses import dataclass, field
from typing import NoReturn, List, Union, Any
from datetime import timedelta

import numpy as np
from numpy import ndarray, where


@dataclass
class Place:
    tokens: int
    meaning: str
    identifier: str
    is_control_place: bool = False
    inputs: List = field(default_factory=list)
    outputs: List = field(default_factory=list)

    def update(self, tokens):
        """
        update number of tokens in place
        Returns:

        """
        self.tokens = tokens
        return self.tokens


@dataclass
class Transition:
    identifier: str
    intrinsic_time: timedelta
    meaning: str
    callback: callable = field(default=lambda: print("Activated!"))
    inputs: List = field(default_factory=list)
    outputs: List = field(default_factory=list)
    input_places: List[Place] = field(default_factory=list)  # Is it necessary?
    output_places: List[Place] = field(default_factory=list)  # Is it necessary?

    @property
    def is_allowed(self) -> bool:
        ...

    def vivacity(self):
        ...

    def __gt__(self, other):
        return self.identifier > other.identifier

    def __lt__(self, other):
        return self.identifier < other.identifier

    def __eq__(self, other):
        return self.identifier == other.identifier


@dataclass
class Arc:
    input: Union[Place, Transition]
    output: Union[Place, Transition]
    weight: float = 1

    def __post_init__(self):
        self.input.outputs.append(self.output)
        self.output.inputs.append(self.input)

    def __repr__(self):
        return str((self.input.identifier, self.output.identifier))


@dataclass
class MarkingNode:
    marking: ndarray
    name: str
    father: Any = None
    transition: Transition = None
    childrens: List = field(default_factory=list)
    hashable_name: str = field(init=False)
    is_terminal: bool = False

    def __post_init__(self):
        self.hashable_name = ' '.join(str(int(k)) if not np.isinf(k) else 'W' for k in self.marking)

    def __str__(self):
        return self.hashable_name
    __repr__ = __str__

    def is_covered_by(self, marking):
        return all(marking.marking >= self.marking)

    def is_dominant_over(self, marking):
        return marking.is_covered_by(self) and any(self.marking > marking.marking)

    def set_omega(self, indexes):
        self.marking[indexes] = np.Inf
        self.hashable_name = ' '.join(
            str(int(k)) if i not in indexes and not np.isinf(k) else 'W'
            for i, k in enumerate(self.marking)
        )

    def check_dominance(self):
        current_node = self.father
        while current_node is not None:
            if self.is_dominant_over(current_node):
                bigger = where(self.marking > current_node.marking)[0]
                self.set_omega(bigger)
                break
            else:
                current_node = current_node.father



