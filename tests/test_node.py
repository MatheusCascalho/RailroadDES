from models.node import Node
from datetime import timedelta


def test_node_petri_model():
    node = Node(
        queue_capacity=3,
        name="NEX",
        slots=1,
        process_time=timedelta(),
        initial_trains={'receiving': 1}
    )
    node.petri_model.coverage_tree()
    assert False