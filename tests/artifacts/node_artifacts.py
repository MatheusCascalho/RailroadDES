import json

import pytest
from src.domain.entities.model_queue import Queue
from src.domain.constants import Process
from src.domain.entities.node import Node
from faker import Faker
from factory import Factory, LazyAttribute
from src.domain.entities.node_data_model import StockData, ProcessorRate, StockNodeData, RateData
from faker_food import FoodProvider
from src.domain.entities.railroad_mesh import RailroadMesh
from src.domain.factories.node_factory import NodeStockFactory
from src.simulation.clock import Clock
from datetime import datetime, timedelta
from src.domain.entities.railroad_mesh import TransitTime


@pytest.fixture
def mock_train(mocker):
    train = mocker.Mock()
    train.ID = "T1"
    train.current_process_name = Process.LOAD
    return train


@pytest.fixture
def mock_simulator(mocker):
    simulator = mocker.Mock()
    simulator.current_date = "2025-01-01"
    return simulator


@pytest.fixture
def mock_slot(mocker):
    slot = mocker.Mock()
    slot.type = Process.LOAD
    slot.is_idle = True
    slot.get_process_time.return_value = 5
    return slot


@pytest.fixture
def basic_node(mocker):
    clock = mocker.Mock()
    clock.current_time = "12:00"
    return Node(
        name="N1",
        clock=clock,
        process_constraints=[],
        maneuvering_constraint_factory=mocker.Mock(),
        queue_to_enter=mocker.Mock(),
        queue_to_leave=mocker.Mock(),
        process_units=[]
    )

@pytest.fixture
def basic_node_factory(mocker):
    def make(name, clock=None):
        clk = mocker.Mock() if not clock else clock
        if not clock:
            clk.current_time = "12:00"
        return Node(
            name=name,
            clock=clk,
            process_constraints=[],
            maneuvering_constraint_factory=mocker.Mock(),
            queue_to_enter=mocker.Mock(),
            queue_to_leave=mocker.Mock(),
            process_units=[]
        )
    return make

@pytest.fixture
def real_node_factory(mocker):
    def make(name, clock=None):
        clk = mocker.Mock() if not clock else clock
        if not clock:
            clk.current_time = "12:00"
        return Node(
            name=name,
            clock=clk,
            process_constraints=[],
            maneuvering_constraint_factory=mocker.Mock(),
            queue_to_enter=Queue(capacity=20, name='input'),
            queue_to_leave=Queue(capacity=20, name='output'),
            process_units=[]
        )
    return make


fake = Faker('pt_BR')  # Definindo o locale para português do Brasil
fake.add_provider(FoodProvider)

class StockFactory(Factory):
    class Meta:
        model = StockData

    product = LazyAttribute(lambda x: fake.vegetable())
    capacity = LazyAttribute(
        lambda x: max(fake.random_number(digits=5),10_000)#, 10.00)  # Desconto mínimo de 10.00
    )
    initial_volume = LazyAttribute(
        lambda obj: min(fake.random_number(digits=5),obj.capacity)
    )


class RateDataFactory(Factory):
    class Meta:
        model = RateData

    product: str
    rate = fake.random_number(digits=4)


class ProcessorRateFactory(Factory):
    class Meta:
        model = ProcessorRate
    product: str
    type: Process
    rate = fake.random_number(digits=4)

class StockNodeDataFactory(Factory):
    class Meta:
        model = StockNodeData

    name: str = "xpto"
    stocks: list[StockData] = [StockFactory() for _ in range(5)]
    rates: list[ProcessorRate] = []
    replenishment: list[dict] = []
    train_sizes: list[float] = []
    queue_capacity: int = 20
    post_operation_time: int = fake.random_number(digits=2)/10

    @classmethod
    def _create(cls, model_class, *args, **kwargs):
        process = kwargs.pop('process', None)  # Pega o parâmetro extra
        obj = super()._create(model_class, *args, **kwargs)

        # Agora você pode manipular o objeto ou fazer algo com o extra_param
        cls.rates = [ProcessorRateFactory(product=stock.product, type=process) for stock in obj.stocks]
        # if extra_param:
        #     print(f"Valor extra recebido: {extra_param}")

        return obj

if __name__ == '__main__':
    import json

    with open('transit_times.json', 'r') as f:
        transits = json.load(f)
    nodes = []
    load_points = []
    unload_points = []
    clk = Clock(start=datetime(2025, 4, 1), discretization=timedelta(hours=1))
    for tt in transits:
        origin = StockNodeDataFactory(name=tt['load_origin'], process=Process.LOAD)
        destination = StockNodeDataFactory(name=tt['load_destination'], process=Process.UNLOAD)
        load_points.append(NodeStockFactory(data=origin, clock=clk).create_node())
        unload_points.append(NodeStockFactory(data=destination, clock=clk).create_node())
    mesh = RailroadMesh(
        load_points=tuple(load_points),
        unload_points=tuple(unload_points),
        transit_times=[TransitTime(**t) for t in transits]
    )
    print(mesh)


