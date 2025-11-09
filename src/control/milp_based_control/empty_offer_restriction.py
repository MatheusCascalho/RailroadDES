import numpy as np
from src.control.milp_based_control.railroad_elements import Flow, RailroadProblemTemplate
from src.control.milp_based_control.constraints.restrictions import Restrictions, RestrictionType, Restriction


class EmptyOfferRestriction(RailroadProblemTemplate, Restrictions):
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
        for i, unload_point in enumerate(self.empty_origins):
            matrix = np.zeros(self.cardinality)
            matrix[:, i, :, :] += 1     # departure trains
            matrix[:, :, :, i] += -1    # arrived trains
            restriction = Restriction(
                coefficients=matrix,
                sense=self.restriction_type.value,
                resource=unload_point.initial_trains
            )
            restrictions.append(restriction)
        return restrictions

    def restrictions(self) -> list[Restriction]:
        return self.__restrictions

    @property
    def restriction_type(self) -> RestrictionType:
        return RestrictionType.LESS_OR_EQUAL

