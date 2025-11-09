from datetime import timedelta

from src.railroad_mesh import RailSegment
from pytest import fixture

@fixture
def simple_rail_segment(simple_stock_node_factory):
    origin = simple_stock_node_factory('origin')
    destination = simple_stock_node_factory('destination')
    segment = RailSegment(
        origin=origin,
        destination=destination,
        time_to_origin=timedelta(hours=5),
        time_to_destination=timedelta(hours=5)
    )
    return segment
