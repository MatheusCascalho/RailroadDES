import numpy as np
from src.railroad_elements import Flow, RailroadProblemTemplate
from src.restrictions import Restrictions, RestrictionType, Restriction


class CapacityRestrictions(RailroadProblemTemplate, Restrictions):
    def __init__(
            self,
            trains: int,
            flows: list[Flow]
    ):
        super().__init__(
            trains=trains,
            flows=flows
        )
        self.__restrictions = self.__build_restrictions(flows=flows)

    def __build_restrictions(self, flows):
        restrictions = []
        for j, origin in enumerate(self.loaded_origins):
            matrix = np.zeros(self.cardinality)
            filtered_flows = [flow for flow in flows if flow.origin == origin]
            if filtered_flows:
                for flow in filtered_flows:
                    coefficient = flow.train_volume
                    k = self.loaded_destinations.index(flow.destination)
                    matrix[:, :, j, k] = coefficient
                restriction = Restriction(
                    coefficients=matrix,
                    sense=self.restriction_type.value,
                    resource=origin.capacity
                )
                restrictions.append(restriction)
        return restrictions

    def restrictions(self) -> list[Restriction]:
        return self.__restrictions

    @property
    def restriction_type(self) -> RestrictionType:
        return RestrictionType.LESS_OR_EQUAL

