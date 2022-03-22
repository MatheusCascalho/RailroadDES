from enum import IntEnum


EPSILON = 1e-1


class TrainActions:
    MOVING = 0
    LOADING = 1
    UNLOADING = 2
    MANEUVERING_TO_ENTER = 3
    MANEUVERING_TO_LEAVE = 4
    WAITING_ON_QUEUE_TO_ENTER = 5
    WAITING_ON_QUEUE_TO_LEAVE = 6

