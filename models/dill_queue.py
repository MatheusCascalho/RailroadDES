import multiprocessing
import dill

class DillQueue:
    def __init__(self, name, maxsize=10_000):
        self.name = name
        self.queue = multiprocessing.Queue(maxsize=maxsize)
        self.persistent_memory = []

    def put(self, item):
        self.queue.put(dill.dumps(item))
        self.persistent_memory.append(item)
        pass

    def get(self, timeout=1):
        data = self.queue.get(timeout=timeout)
        return dill.loads(data)

    def empty(self):
        return self.queue.empty()

    def qsize(self):
        return self.queue.qsize()
