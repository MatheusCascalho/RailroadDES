from models.demand import Demand
from models.observers import AbstractSubject, to_notify
from models.time_table import TimeTable, TimeEvent
from models.constants import Process, EventName
from models.path import Path
from datetime import datetime
from typing import Any


def task_id_gen():
    i = 0
    while True:
        name = f"task_{i}"
        yield name
        i += 1

task_id = task_id_gen()


class Task(AbstractSubject):
    def __init__(
            self,
            demand: Demand,
            path: list[str],
            task_volume: float,
            current_time: datetime,
            state: Any
    ):
        """
        Initializes a Task object that represents a routing decision for a train at a specific point in the simulation.

        Args:
            demand (Demand): The demand associated with this task.
            task_volume (float): The total volume of the task.
            current_time (datetime): The current time of the simulation at which the task is registered.
        """
        super().__init__()
        self.ID = next(task_id)
        self.demand = demand
        demand.promised += task_volume
        self.path = Path(path)
        self.time_table = TimeTable()
        event = TimeEvent(
            event=EventName.DEPARTURE,
            instant=current_time
        )
        self.time_table.update(event=event, process=Process.UNLOAD)
        self.task_volume = task_volume
        self.invoiced_volume = 0
        self.invoiced_volume_time = None
        self.train_id = None
        self.model_state = state

    def update(self, event: TimeEvent, process: Process, location: str=None):
        """
        Updates the task's timetable and the invoiced volume based on the event.

        This method updates the `TimeTable` with the provided event and process. If the event is `FINISH_PROCESS`
        and the current process is `LOAD`, it updates the invoiced volume to the task's volume.

        Args:
            event (TimeEvent): The event to be updated in the task's timetable.
            process (Process): The process related to the event.
        """
        self.time_table.update(event, process=process, location=location)
        should_update_invoiced_volume = (
                event.event == EventName.FINISH_PROCESS and
                self.time_table.current_process == Process.LOAD
        )
        if should_update_invoiced_volume:
            self.update_invoiced_volume(event)

    @to_notify()
    def update_invoiced_volume(self, event: TimeEvent):
        self.invoiced_volume = self.task_volume
        self.demand.operated += self.invoiced_volume
        self.invoiced_volume_time = event.instant

    def assign(self, train_id: str):
        self.train_id = train_id

    def penalty(self):
        """
        Calculates the penalty of the task, which is based on the queue time in the timetable.

        Returns:
            timedelta: The total penalty time, which is the queue time in the timetable.
        """

        queue = self.time_table.queue_time
        return queue

    def reward(self):
        """
        Returns the reward associated with the task, which is the invoiced volume.

        Returns:
            float: The invoiced volume for the task.
        """
        return self.invoiced_volume

    def is_on_load_point(self):
        return self.path.current_location.split('-')[0] == self.demand.flow.origin

    def __repr__(self):
        return f"{self.ID} - {self.demand.flow}"
