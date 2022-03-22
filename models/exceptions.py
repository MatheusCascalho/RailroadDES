class TrainExceptions(Exception):
    @staticmethod
    def path_is_finished():
        raise TrainExceptions('Path is finished!')

    @staticmethod
    def volume_to_unload_is_greater_than_current_volume():
        raise TrainExceptions('Volume to unload is greater than current volume!')
