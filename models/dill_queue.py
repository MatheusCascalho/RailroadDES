import multiprocessing
import dill

class DillQueue:
    def __init__(self, maxsize=10_000):
        self.queue = multiprocessing.Queue(maxsize=maxsize)

    def put(self, item):
        self.queue.put(dill.dumps(item))
        pass

    def get(self, timeout=1):
        data = self.queue.get(timeout=timeout)
        return dill.loads(data)

    def empty(self):
        return self.queue.empty()

    def qsize(self):
        return self.queue.qsize()
