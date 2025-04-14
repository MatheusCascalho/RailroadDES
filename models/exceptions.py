from interfaces.train_interface import TrainInterface
from datetime import datetime


class FinishedTravelException(Exception):
    def __init__(self, value, train: TrainInterface, current_time: datetime):
        self.current_time = current_time
        self.value = value
        self.train = train

    @staticmethod
    def path_is_finished(train: TrainInterface, current_time):
        raise FinishedTravelException('Travel is finished!', train=train, current_time=current_time)


class TrainExceptions(Exception):
    @staticmethod
    def path_is_finished():
        raise TrainExceptions('Path is finished!')

    @staticmethod
    def volume_to_unload_is_greater_than_current_volume():
        raise TrainExceptions('Volume to unload is greater than current volume!')

    @staticmethod
    def processing_when_train_is_moving():
        raise TrainExceptions('Train could not be processed if it is moving state. Arrives train at node first!')


class EventSequenceError(Exception):
    """Exceção customizada para erro na sequência de eventos."""

    def __init__(self, message="The event sequence is incorrect."):
        self.message = message
        super().__init__(self.message)


class TimeSequenceErro(Exception):
    """Exceção customizada para erro na sequência de eventos."""

    def __init__(self, message="The current event is before the last event"):
        self.message = message
        super().__init__(self.message)


class RepeatedProcessError(Exception):
    """Exceção customizada para erro de repetição de processo."""

    def __init__(self, message="Process is repeated."):
        self.message = message
        super().__init__(self.message)


class AlreadyRegisteredError(Exception):
    """Exceção customizada para erro na sequência de eventos."""

    def __init__(self, message="Event already registered"):
        self.message = message
        super().__init__(self.message)
