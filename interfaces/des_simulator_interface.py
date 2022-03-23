class DESSimulatorInterface:
    """
    Discrete Event System simulator
    """
    def add_event(self, time, callback, **data):
        pass

    def simulate(self, model, time_horizon=28 * 3600):
        pass
