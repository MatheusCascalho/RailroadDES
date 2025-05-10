from interfaces.train_interface import TrainInterface
from datetime import datetime

class NotCompletedEvent(Exception):
    def __init__(self, event=None, message='Event not completed'):
        super().__init__(message)

class FinishedTravelException(Exception):
    def __init__(self, value, train: TrainInterface, current_time: datetime):
        self.current_time = current_time
        self.value = value
        self.train = train

    @staticmethod
    def path_is_finished(train: TrainInterface, current_time):
        raise FinishedTravelException('Travel is finished!', train=train, current_time=current_time)


class TrainExceptions(Exception):
    def __init__(self, message, train_id: str, location:str):
        super().__init__(message)
        self.train_id = train_id
        self.location = location

    @staticmethod
    def train_is_not_in_slot(train_id: str, location:str):
        raise TrainExceptions("Train is not in slot!", train_id, location)

    @staticmethod
    def path_is_finished(train_id: str, location:str):
        raise TrainExceptions('Path is finished!', train_id, location)

    @staticmethod
    def volume_to_unload_is_greater_than_current_volume():
        raise TrainExceptions('Volume to unload is greater than current volume!')

    @staticmethod
    def processing_when_train_is_moving():
        raise TrainExceptions('Train could not be processed if it is moving state. Arrives train at node first!')


class StockException(Exception):
    @staticmethod
    def stock_is_full():
        raise StockException("Stock is full!")

    @staticmethod
    def stock_is_empty():
        raise StockException("Stock is Empty")

    @staticmethod
    def no_receive_event():
        raise StockException("Stock has no receive event!")


class ProcessException(Exception):
    @staticmethod
    def no_promise_to_do():
        raise ProcessException("There is no promise to do!")

    @staticmethod
    def process_is_busy():
        raise ProcessException("Processor is Busy")

    @staticmethod
    def process_is_blocked():
        raise ProcessException("Processor is Blocked")


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

    def __init__(self, message="Event already registered with another instant"):
        self.message = message
        super().__init__(self.message)
