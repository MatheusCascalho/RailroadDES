from enum import IntEnum, Enum


EPSILON = 1e-1


class TrainActions(IntEnum):
    MOVING = 0
    LOADING = 1
    UNLOADING = 2
    MANEUVERING_TO_ENTER = 3
    MANEUVERING_TO_LEAVE = 4
    WAITING_ON_QUEUE_TO_ENTER = 5
    WAITING_ON_QUEUE_TO_LEAVE = 6


class Process(Enum):
    LOAD = 'load'
    UNLOAD = 'unload'


class EventName(Enum):
    """
    Enum that defines the possible event types in the simulation process.
    """
    ARRIVE = "Arrive"
    START_PROCESS = "Start Process"
    FINISH_PROCESS = "Finish Process"
    DEPARTURE = "Departure"
    RECEIVE_VOLUME = "Receive Volume in Stock"
    DISPATCH_VOLUME = "Dispatch Volume in Stock"
