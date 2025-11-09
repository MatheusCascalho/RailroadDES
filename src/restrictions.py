"""
This file declares the restriction interface - class Restriction - and implements each restriction for
railroad optimization problem - ROP

"""

from abc import abstractmethod, ABC
import numpy as np
from dataclasses import dataclass
from enum import Enum


class RestrictionType(Enum):
    EQUALITY = "="
    LESS_OR_EQUAL = "<"
    GREATER_OR_EQUAL = ">"
    INEQUALITY = "<>"


@dataclass
class Restriction:
    coefficients: np.ndarray
    sense: str
    resource: float

    def __vectorize(self, data):
        for dimension in data:
            if isinstance(dimension, np.ndarray):
                for d in self.__vectorize(dimension):
                    yield d
            else:
                yield dimension

    def to_vector(self):
        data = list(self.__vectorize(self.coefficients))
        data = np.array(data)
        return data


class Restrictions(ABC):
    @abstractmethod
    def restrictions(self) -> list[Restriction]:
        pass

    @abstractmethod
    def restriction_type(self) -> RestrictionType:
        pass

    @property
    def coefficients_matrix(self) -> np.ndarray:
        """
        This method build constraint coefficients as a Matrix
        :return:
        """
        a = [r.to_vector() for r in self.restrictions()]
        a = np.array(a)
        return a

    @property
    def resource_vector(self) -> np.ndarray:
        b = [r.resource for r in self.restrictions()]
        b = np.array(b)
        return b

    @abstractmethod
    def cardinality(self):
        pass