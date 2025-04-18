from abc import abstractmethod
from typing import Callable






class AbstractObserver:
    """
    Classe abstrata para implementações de observadores. Os observadores são notificados quando um estado é alterado.
    """
    _subjects: list = []
    def __init__(self):
        self._subjects = []

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, valor):
        self._subjects = valor

    def append_subject(self, sub):
        self.subjects.append(sub)

    @abstractmethod
    def update(self, *args):
        """
        Abstract method that must be implemented to handle state change notifications.        :param args:
        :return:
        """
        pass


class AbstractSubject:
    observers: list[AbstractObserver] = []


    def add_observers(self, observers: list[AbstractObserver]):
        if isinstance(observers, AbstractObserver):
            observers = [observers]
        observers_to_add = [o for o in observers if o not in self.observers]
        AbstractSubject.observers.extend(observers_to_add)
        for obs in observers:
            if self not in obs.subjects:
                obs.append_subject(self)


    @classmethod
    def notify(cls):
        """
        Notifica todos os observadores associados ao estado.
        :return:
        """
        for observer in cls.observers:
            observer.update()

    @staticmethod
    def notify_at_the_end(func: Callable):
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            AbstractSubject.notify()
        return wrapper


def id_gen():
    i = 0
    while True:
        yield i
        i += 1


id_fofoqueiro = id_gen()


class Fofoqueiro(AbstractObserver):
    def __init__(self):
        self.ID = next(id_fofoqueiro)
        super().__init__()
    def update(self, *args):
        print(f"Fofoqueiro {self.ID}: Fui notificado!! Tenho: {len(self.subjects)} Sujeito(s)")
        for sub in self.subjects:
            print(f"{sub.volume=}")
    def __eq__(self, other):
        return self.ID == other.ID
    def __str__(self):
        return f"Fofoqueiro {self.ID}"
    __repr__ = __str__
