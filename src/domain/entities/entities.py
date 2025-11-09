"""
Entidades do sistema.

Uma entidade é um objeto que realiza/sofre eventos. Toda entidade possui um Estado
"""

from dataclasses import dataclass, field, InitVar
from typing import Any, Generator, Callable



@dataclass
class Entity:
    state: Any = field(init=False)

