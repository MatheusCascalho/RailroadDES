from abc import abstractmethod
import types


class AbstractObserver:
    """
    Classe abstrata para implementações de observadores. Os observadores são notificados quando um estado é alterado.
    """
    def __init__(self):
        self._subjects: list[AbstractSubject] = []

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, valor):
        self._subjects = valor

    def append_subject(self, sub):
        if not isinstance(sub, AbstractSubject):
            raise Exception("Only objects of type 'Subject' can be added to the observer")
        self.subjects.append(sub)

    @abstractmethod
    def update(self, *args):
        """
        Abstract method that must be implemented to handle state change notifications.
        :param args:
        :return:
        """
        pass


class SubjectNotifier:
    def notify_at_the_end(self, observers):
        """
        Decorator to notify subject observers after executing some functionality
        :param observers:
        :return:
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                func(*args, **kwargs)
                self.notify(observers)
            return wrapper
        return decorator

    @staticmethod
    def notify(observers):
        """
        Notifies all observers associated with the subject.
        :return:
        """
        for observer in observers:
            observer.update()


def to_notify(decorator_name="notify_at_the_end"):
    """
    Decorator to identify that the method should be decorated by a notification decorator.
    By default, we will use the `notify_at_the_end` decorator
    :param decorator_name:
    :return:
    """
    def marker(func):
        func._should_notify = decorator_name
        return func
    return marker


class SubjectMetaDecorator(type):
    """Metaclass to decorate methods marked for notification after instance creation"""
    def __new__(cls, name, bases, dct):
        """
        It is called when the class is created, before any instance.
        The goal here is to override the __init__ method so that when the subject is instantiated,
        all methods marked for notification are decorated with the respective decoration method.
        :param args:
        :param kwargs:
        """
        original_init = dct.get("__init__")
        def new_init(self, *args, **kwargs):
            notifier = SubjectNotifier()
            if original_init:
                original_init(self, *args, **kwargs)

            for attr_name in dir(self):
                method = getattr(self, attr_name)
                if isinstance(method, types.MethodType):
                    decorator_name = getattr(method, "_should_notify", None)
                    if decorator_name:
                        decorator_func = getattr(notifier, decorator_name)
                        decorated = decorator_func(self.observers)(method)
                        setattr(self, attr_name, decorated)

        dct['__init__'] = new_init
        return super().__new__(cls, name, bases, dct)


class AbstractSubject(metaclass=SubjectMetaDecorator):
    def __init__(self):
        self.observers: list[AbstractObserver] = []

    def add_observers(self, observers: list[AbstractObserver]):
        """
        Adds an observer to the subject and the subject itself to the observer.
        :param observers:
        :return:
        """
        if isinstance(observers, AbstractObserver):
            observers = [observers]
        observers_to_add = [o for o in observers if o not in self.observers]
        self.observers.extend(observers_to_add)
        for obs in observers:
            if self not in obs.subjects:
                obs.append_subject(self)

class SubjectWithOnlyOneObserver(metaclass=SubjectMetaDecorator):
    def __init__(self):
        self.observers: list[AbstractObserver] = []

    def add_observers(self, observers: list[AbstractObserver]):
        """
        Adds an observer to the subject and the subject itself to the observer.
        :param observers:
        :return:
        """
        if isinstance(observers, AbstractObserver):
            observers = [observers]
        if len(observers)> 1:
            raise Exception("This subject should be observed by only one object")
        self.observers.extend(observers)
        while len(self.observers)>1:
            self.observers.pop(0)
        if self not in observers[0].subjects:
            observers[0].append_subject(self)


def id_gen():
    i = 0
    while True:
        yield i
        i += 1


id_gossiper = id_gen()


class Gossiper(AbstractObserver):
    def __init__(self):
        self.ID = next(id_gossiper)
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
