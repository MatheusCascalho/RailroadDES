import pytest
from unittest.mock import MagicMock

from models.observers import (
    AbstractObserver,
    AbstractSubject,
    SubjectNotifier,
    to_notify,
    SubjectMetaDecorator
)

# ============================================
# Mocks
# ============================================

class MockObserver(AbstractObserver):
    def update(self, *args):
        self.updated = True

class MockSubject(AbstractSubject, metaclass=SubjectMetaDecorator):
    def __init__(self):
        super().__init__()

    @to_notify()
    def some_method(self):
        self.executed = True


# ============================================
# Testes AbstractObserver
# ============================================

def test_append_valid_subject():
    observer = MockObserver()
    subject = AbstractSubject()

    observer.append_subject(subject)

    assert subject in observer.subjects

def test_append_invalid_subject():
    observer = MockObserver()

    with pytest.raises(Exception, match="Only objects of type 'Subject' can be added"):
        observer.append_subject("not_a_subject")

def test_subjects_getter_setter():
    observer = MockObserver()
    subjects = [AbstractSubject(), AbstractSubject()]

    observer.subjects = subjects
    assert observer.subjects == subjects

# ============================================
# Testes SubjectNotifier
# ============================================

def test_notify_calls_update():
    observer1 = MockObserver()
    observer2 = MockObserver()
    observer1.update = MagicMock()
    observer2.update = MagicMock()

    SubjectNotifier.notify([observer1, observer2])

    observer1.update.assert_called_once()
    observer2.update.assert_called_once()

def test_notify_at_the_end_calls_notify():
    notifier = SubjectNotifier()
    observer = MockObserver()
    observer.update = MagicMock()

    @notifier.notify_at_the_end([observer])
    def dummy_function():
        return "executed"

    result = dummy_function()

    assert result is None  # função original não retorna nada
    observer.update.assert_called_once()

# ============================================
# Testes to_notify
# ============================================

def test_to_notify_marks_function():
    @to_notify("notify_at_the_end")
    def some_func():
        pass

    assert hasattr(some_func, "_should_notify")
    assert some_func._should_notify == "notify_at_the_end"

# ============================================
# Testes SubjectMetaDecorator
# ============================================

def test_subject_meta_decorator_applies_decorator():
    mock_subject = MockSubject()
    observer = MockObserver()

    mock_subject.add_observers(observer)

    observer.update = MagicMock()

    mock_subject.some_method()

    # Depois de chamar o método decorado, o observer deve ter sido notificado
    observer.update.assert_called_once()

# ============================================
# Testes AbstractSubject
# ============================================

def test_add_single_observer():
    subject = AbstractSubject()
    observer = MockObserver()

    subject.add_observers(observer)

    assert observer in subject.observers
    assert subject in observer.subjects

def test_add_multiple_observers():
    subject = AbstractSubject()
    observer1 = MockObserver()
    observer2 = MockObserver()

    subject.add_observers([observer1, observer2])

    assert observer1 in subject.observers
    assert observer2 in subject.observers
    assert subject in observer1.subjects
    assert subject in observer2.subjects

def test_no_duplicate_observers():
    subject = AbstractSubject()
    observer = MockObserver()

    subject.add_observers(observer)
    subject.add_observers(observer)  # adicionar novamente

    assert subject.observers.count(observer) == 1
