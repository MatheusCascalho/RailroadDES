import numpy as np
from src.railroad_elements import Flow, Demand, RailroadProblemTemplate
from src.restrictions import Restrictions, RestrictionType, Restriction


class MinimumDemandRestriction(RailroadProblemTemplate, Restrictions):
    def __init__(
            self,
            trains,
            demands: list[Demand],
    ):
        super().__init__(
            trains=trains,
            flows=[d.flow for d in demands]
        )
        self.__restrictions = self.__build_restrictions(demands=demands)

    def __build_restrictions(self, demands: list[Demand]) -> list[Restriction]:
        restrictions = []
        for j, origin in enumerate(self.loaded_origins):
            for k, destination in enumerate(self.loaded_destinations):
                matrix = np.zeros(self.cardinality)
                filtered_bands = [
                    d
                    for d in demands
                    if d.flow.origin == origin and
                       d.flow.destination == destination
                ]
                if filtered_bands and filtered_bands[0].minimum:
                    demand = filtered_bands[0]
                    matrix[:, :, j, k] = demand.flow.train_volume
                    restriction = Restriction(
                        coefficients=matrix,
                        sense=self.restriction_type.value,
                        resource=demand.minimum
                    )
                    restrictions.append(restriction)
        return restrictions

    def restrictions(self) -> list[Restriction]:
        return self.__restrictions

    @property
    def restriction_type(self) -> RestrictionType:
        return RestrictionType.GREATER_OR_EQUAL


class MaximumDemandRestriction(RailroadProblemTemplate, Restrictions):
    def __init__(
            self,
            trains,
            demands: list[Demand]
    ):
        super().__init__(
            trains=trains,
            flows=[d.flow for d in demands]
        )
        self.__restrictions = self.__build_restrictions(demands=demands)

    def __build_restrictions(self, demands: list[Demand]) -> list[Restriction]:
        restrictions = []
        for j, origin in enumerate(self.loaded_origins):
            for k, destination in enumerate(self.loaded_destinations):
                matrix = np.zeros(self.cardinality)
                filtered_bands = [
                    d
                    for d in demands
                    if d.flow.origin == origin and
                       d.flow.destination == destination
                ]
                if filtered_bands:
                    demand = filtered_bands[0]
                    matrix[:, :, j, k] = demand.flow.train_volume
                    restriction = Restriction(
                        coefficients=matrix,
                        sense=self.restriction_type.value,
                        resource=demand.maximum
                    )
                    restrictions.append(restriction)
        return restrictions

    def restrictions(self) -> list[Restriction]:
        return self.__restrictions

    @property
    def restriction_type(self) -> RestrictionType:
        return RestrictionType.LESS_OR_EQUAL
