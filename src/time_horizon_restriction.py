import numpy as np
from src.railroad_elements import Flow, TransitTime, RailroadProblemTemplate
from src.restrictions import Restrictions, RestrictionType, Restriction


class TimeHorizonRestriction(RailroadProblemTemplate, Restrictions):
    def __init__(
            self,
            trains,
            transit_times: list[TransitTime],
            flows: list[Flow],
            time_horizon: int
    ):
        super().__init__(
            trains=trains,
            flows=flows
        )
        self.__restrictions = self.__build_restrictions(
            transit_times=transit_times,
            time_horizon=time_horizon,
            flows=flows
        )

    def __build_restrictions(
            self,
            transit_times: list[TransitTime],
            time_horizon: int,
            flows: list[Flow]
    ) -> list[Restriction]:
        restrictions = []
        transit_matrix = self.__build_transit_matrix(transit_times=transit_times)
        train_volumes = self.__build_train_volumes(flows=flows)
        self.transit_matrix = transit_matrix
        for n in range(self.cardinality[0]):
            matrix = np.zeros(self.cardinality)
            for j, load_point in enumerate(self.loaded_origins):
                for k, unload_destination in enumerate(self.loaded_destinations):
                    time = transit_matrix[j, k]
                    train_volume = train_volumes[j, k]
                    load_rate = load_point.capacity / time_horizon
                    load_process = train_volume / load_rate
                    unload_rate = unload_destination.capacity / time_horizon
                    unload_process = train_volume / unload_rate
                    matrix[n, :, j, k] += time + unload_process
                    matrix[n, k, j, :] += time + load_process
            restriction = Restriction(
                coefficients=matrix,
                sense=self.restriction_type.value,
                resource=time_horizon
            )
            restrictions.append(restriction)
        return restrictions

    def __build_transit_matrix(self, transit_times: list[TransitTime]) -> np.ndarray:
        l = len(self.loaded_origins)
        u = len(self.loaded_destinations)
        shape = (l, u)
        matrix = np.zeros(shape)
        for transit in transit_times:
            if transit.origin in self.loaded_origins:
                j = self.loaded_origins.index(transit.origin)
                k = self.loaded_destinations.index(transit.destination)
                matrix[j, k] = transit.time
        return matrix

    def __build_train_volumes(self, flows):
        l = len(self.loaded_origins)
        u = len(self.loaded_destinations)
        shape = (l, u)
        matrix = np.zeros(shape)
        for flow in flows:
            i = self.loaded_origins.index(flow.origin)
            j = self.loaded_destinations.index(flow.destination)
            matrix[i, j] = flow.train_volume
        return matrix

    def restrictions(self) -> list[Restriction]:
        return self.__restrictions

    @property
    def restriction_type(self) -> RestrictionType:
        return RestrictionType.LESS_OR_EQUAL
