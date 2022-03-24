from interfaces.train_interface import TrainInterface


class FinishedTravelException(Exception):
    def __init__(self, value, train: TrainInterface):
        self.value = value
        self.train = train

    @staticmethod
    def path_is_finished(train: TrainInterface):
        raise FinishedTravelException('Travel is finished!', train=train)


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
