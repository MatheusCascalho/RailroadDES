from dataclasses import dataclass, field
from datetime import datetime, timedelta
from models.constants import Process, EventName
from models.observers import AbstractSubject, to_notify
import numpy as np
from models.exceptions import EventSequenceError, TimeSequenceErro, RepeatedProcessError, AlreadyRegisteredError


@dataclass
class TimeEvent:
    """
    Represents a time event in the simulation, with an associated event type and timestamp.

    Attributes:
    event (EventName): The type of the event (ARRIVE, START_PROCESS, FINISH_PROCESS, or DEPARTURE).
    instant (datetime): The timestamp when the event occurs.
    """
    event: EventName
    instant: datetime = field(default=None)

    def __eq__(self, other):
        if isinstance(other, datetime):
            return self.instant == other
        elif isinstance(other, TimeEvent):
            return self.instant == other.instant and self.event == other.event

    def __add__(self, other):
        if isinstance(other, datetime):
            return self.instant + other
        elif isinstance(other, TimeEvent):
            return self.instant + other.instant

    def __sub__(self, other):
        if isinstance(other, datetime):
            return self.instant - other
        elif isinstance(other, TimeEvent):
            return self.instant - other.instant


def register_id_gen():
    i = 0
    while True:
        name = f"register_{i}"
        yield name
        i += 1


register_id = register_id_gen()


@dataclass
class TimeRegister:
    """
    Represents a time register for a process during the simulation. Stores timestamps for different events
    like arrival, start of process, finish of process, and departure.

    Attributes:
    process (Process): The process related to the item.
    arrive (datetime): The arrival time.
    start_process (datetime): The start time of the process.
    finish_process (datetime): The finish time of the process.
    departure (datetime): The departure time.
    """
    process: Process = field(default=None)
    arrive: TimeEvent = field(default=None)
    start_process: TimeEvent = field(default=None)
    finish_process: TimeEvent = field(default=None)
    departure: TimeEvent = field(default=None)
    sequence: tuple[EventName] = (
        EventName.ARRIVE,
        EventName.START_PROCESS,
        EventName.FINISH_PROCESS,
        EventName.DEPARTURE
    )
    event_attr: dict[EventName, TimeEvent] = field(default=dict)
    ID: str = field(init=False)

    def __str__(self):
        times = (f"ARRIVE: {self.arrive.instant} | START: {self.start_process.instant} "
                 f"| FINISH: {self.finish_process.instant} | DEPARTURE: {self.departure.instant}.instant")
        return f"{self.ID} | {self.process} | {times}"

    __repr__ = __str__

    def __post_init__(self):
        self.ID = next(register_id)
        self.arrive = TimeEvent(EventName.ARRIVE) if self.arrive is None else self.arrive
        self.start_process = TimeEvent(EventName.START_PROCESS) if self.start_process is None else self.start_process
        self.finish_process = TimeEvent(EventName.FINISH_PROCESS) if self.finish_process is None else self.finish_process
        self.departure = TimeEvent(EventName.DEPARTURE) if self.departure is None else self.departure
        self.event_attr = {
            EventName.ARRIVE: self.arrive,
            EventName.START_PROCESS: self.start_process,
            EventName.FINISH_PROCESS: self.finish_process,
            EventName.DEPARTURE: self.departure
        }

    def get_queue(self) -> timedelta:
        """
        Calculates the total queue time, which is the sum of the time from arrival to start of process,
        and from finish of process to departure.

        Returns:
        timedelta: The total queue time.
        """
        queue = timedelta()
        if self.start_process.instant is not None:
            queue += self.start_process - self.arrive
        if (
                self.departure.instant is not None and
                self.finish_process.instant is not None
        ):
            queue += self.departure - self.finish_process
        return queue

    def get_process_time(self) -> timedelta:
        """
        Calculates the process time, which is the difference between finish process and start process.

        Returns:
        timedelta: The process time.
        """
        if self.finish_process is not None:
            return self.finish_process - self.start_process
        return timedelta()

    @property
    def is_initial_register(self):
        """
        Checks whether the register is an initial register. An initial register occurs when the departure is
        not None but arrival, start process, or finish process are None.

        Returns:
        bool: True if the register is initial, False otherwise.
        """
        first_defined = next(
            (self.event_attr[e] for e in self.sequence if self.event_attr[e].instant is not None),
            None
        )
        if first_defined is None:
            return False
        if first_defined.event != EventName.ARRIVE:
            return True
        return False

    def update(self, event: TimeEvent):
        """
        Updates the time register with the event's timestamp.

        Args:
        event (TimeEvent): The event to be updated in the register.

        Raises:
        Exception: If the event has already been registered in the current register.
        """
        if self.event_attr[event.event].instant is not None and self.event_attr[event.event].instant != event.instant:
            raise AlreadyRegisteredError()
        event_index = self.sequence.index(event.event)
        predecessors = self.sequence[:event_index]
        all_predecessor_are_defined = all(self.event_attr[p].instant is not None for p in predecessors)
        is_initial_register = self.is_initial_register or all(self.event_attr[p].instant is None for p in predecessors)
        if all_predecessor_are_defined:
            if predecessors and self.event_attr[predecessors[-1]].instant > event.instant:
                raise TimeSequenceErro()
            if self.event_attr[event.event].instant is None:
                self.event_attr[event.event].instant = event.instant
            else:
                raise AlreadyRegisteredError()
        elif is_initial_register:
            self.event_attr[event.event].instant = event.instant
            if self.__check_sequence(predecessors):
                self.event_attr[event.event] = TimeEvent(event.event, None)
                raise EventSequenceError()
        else:
            raise EventSequenceError()

    def __check_sequence(self, predecessors):
        changes = 0
        for i in range(1, len(predecessors)+1):
            current_is_none = (self.event_attr[self.sequence[i]].instant is None)
            previous_is_none = (self.event_attr[self.sequence[i-1]].instant is None)
            if current_is_none ^ previous_is_none:
                changes += 1
        return changes > 1


class TimeTable(AbstractSubject):
    """
    The main class that manages time registers (`TimeRegister`) during the simulation.
    Each event (arrival, process start, finish, and departure) is recorded and used to calculate
    queue time, transit time, and utilization time.

    Attributes:
    registers (list[TimeRegister]): A list of TimeRegister objects that represent the time records during the simulation.
    """
    def __init__(
            self,
            registers: list[TimeRegister] = None
    ):
        """
        Initializes the TimeTable with an optional list of TimeRegister objects.

        Args:
        registers (list[TimeRegister], optional): A list of TimeRegister objects. Defaults to an empty list if not provided.
        """
        if registers is None:
            registers = []
        self.registers = registers
        self.__last_event = None
        super().__init__()

    def __str__(self):
        return f"Table with {len(self.registers)} registers"

    __repr__ = __str__

    @to_notify()
    def update(self, event: TimeEvent, process: Process = Process.UNLOAD):
        """
        Updates the last TimeRegister with the provided event, or creates a new TimeRegister if necessary.

        Args:
        event (TimeEvent): The event to be recorded in the TimeTable.
        """
        if event == self.__last_event:
            print("Evento já registrado") # TODO: Migrar para log.warning
            return
        if self.current_process is not None and self.current_process != process:
            if event.event != EventName.ARRIVE:
                raise TimeSequenceErro()
            elif self.current_process and self.registers[-1].departure.instant is None:
                raise TimeSequenceErro()
        if self.registers and event.event == EventName.ARRIVE:
            if self.current_process == process and self.registers[-1].arrive != event.instant:
                raise RepeatedProcessError()
            if self.registers[-1].departure.instant is None:
                raise EventSequenceError()
            if event.instant < self.registers[-1].departure.instant:
                raise TimeSequenceErro()
        if not self.registers or event.event == EventName.ARRIVE:
            self.registers.append(TimeRegister(process=process))
        self.registers[-1].update(event)
        self.__last_event = event

    @property
    def queue_time(self) -> timedelta:
        """
        Calculates the total queue time by summing the queue times of all TimeRegister objects.

        Returns:
        timedelta: The total queue time.
        """
        queues = [
            register.get_queue()
            for register in self.registers
        ]
        queue = np.sum(queues)
        return queue

    @property
    def in_transit_time(self) -> timedelta:
        """
        Calculates the total transit time by summing the time between the departure of one register
        and the arrival of the next register.

        Returns:
        timedelta: The total transit time.
        """
        transit = timedelta()
        for previous, current in zip(self.registers, self.registers[1:]):
            transit += current.arrive - previous.departure
        return transit

    @property
    def util_time(self) -> timedelta:
        """
        Calculates the total utilization time, which is the sum of transit time and process times of all registers.

        Returns:
        timedelta: The total utilization time.
        """
        util = self.in_transit_time
        for register in self.registers:
            util += register.get_process_time()
        return util

    @property
    def current_process(self):
        if not self.registers:
            return None
        return self.registers[-1].process

    @property
    def process_end(self):
        if self.registers:
            return self.registers[-1].finish_process.instant

    @property
    def dispatched_just_now(self):
        if self.registers:
            return self.registers[-1].departure.instant is not None
        return False

    @property
    def arrived_right_now(self):
        if self.registers:
            arrived = self.registers[-1].arrive.instant is not None
            dont_start_anything = self.registers[-1].start_process.instant is None
            return arrived and dont_start_anything
        return False

    def last_event(self):
        return self.__last_event