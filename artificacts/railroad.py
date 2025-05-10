import pytest
from models.railroad_mesh import RailroadMesh, TransitTime
from models.node import Node
from datetime import timedelta


@pytest.fixture
def simple_mesh():
    terminal = Node(queue_capacity=1, identifier=0)
    port = Node(queue_capacity=1, identifier=1)
    transit = TransitTime(load_origin=terminal.identifier, load_destination=port.identifier,
                          loaded_time=timedelta(hours=24), empty_time=timedelta(hours=20))
    mesh = RailroadMesh(
        load_points=[terminal],
        unload_points=[port],
        transit_times=[transit]
    )
    return mesh