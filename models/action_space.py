import random

class ActionSpace:
    def __init__(self, demands):
        self.demands = demands + ['AUTOMATIC', 'ROUTING']

    @property
    def n_actions(self):
        return len(self.demands) - 2

    def sample(self):
        i = random.randint(0, len(self.demands) - 3)
        return self.demands[i]

    def to_scalar(self, action):
        if not isinstance(action, str):
            action = str(action)
        flows = [str(d.flow) for d in self.demands[:-2]] + self.demands[-2:]
        v = flows.index(action)
        return v

    def get_demand(self, i):
        if i == len(self.demands) - 1:
            return self.sample()
        return self.demands[i]
    
    @property
    def actions(self):
        return [d for d in self.demands[:-2]]
