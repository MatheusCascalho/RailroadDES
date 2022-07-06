from dataclasses import dataclass
from typing import NoReturn, List, Union
from datetime import timedelta


@dataclass
class Place:
    tokens: int
    meaning: str
    identifier: str

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
    input_places: List[Place]
    output_places: List[Place]
    meaning: str

    @property
    def is_allowed(self) -> bool:
        ...

    def vivacity(self):
        ...


@dataclass
class Arc:
    input: Union[Place, Transition]
    output: Union[Place, Transition]
    weight: float = 1


