import numpy as np
from models.railroad_elements import Flow, RailroadProblemTemplate
from models.restrictions import Restrictions, RestrictionType, Restriction


class DispatchInitialTrain(RailroadProblemTemplate, Restrictions):
    def __init__(
            self,
            trains: int,
            flows: list[Flow]
    ):
        super().__init__(
            trains=trains,
            flows=flows
        )
        self.__restrictions = self.__build_restrictions()

    def __build_restrictions(self):
        """
        restriction: all trains that departure from i need to be arrived in i before or is in initial offer of i
        in other words: departure = arrived + initial -> departure - arrived = initial

        :return:
        """
        restrictions = []
        already_counted_trains = 0
        for i, unload_point in enumerate(self.empty_origins):
            for t in range(unload_point.initial_trains):
                matrix = np.zeros(self.cardinality)
                matrix[t+already_counted_trains, i, :, :] += 1     # departure trains
                restriction = Restriction(
                    coefficients=matrix,
                    sense=self.restriction_type.value,
                    resource=1
                )
                restrictions.append(restriction)
            already_counted_trains += unload_point.initial_trains
        return restrictions

    def restrictions(self) -> list[Restriction]:
        return self.__restrictions

    @property
    def restriction_type(self) -> RestrictionType:
        return RestrictionType.GREATER_OR_EQUAL

