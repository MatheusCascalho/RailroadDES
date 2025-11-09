from src.domain.entities.railroad_mesh import TransitTime, RailroadMesh
from src.domain.entities.railroad import Railroad
from faker import Faker
from factory import Factory, LazyAttribute
import json
from pytest import fixture
from src.domain.entities.demand import Demand, Flow
from src.control.router import RandomRouter
from random import randint

# Inicializando o Faker fora da Factory
fake = Faker('pt_BR')  # Definindo o locale para português do Brasil

# Criando uma Factory para gerar floats aleatórios com valor mínimo
class ProdutoFactory(Factory):
    class Meta:
        model = dict  # Usando um dict como exemplo, mas pode ser qualquer modelo

    preco = LazyAttribute(
        lambda x: max(fake.random_number(digits=5), 50000)  # Preço mínimo de 50.000
    )
    desconto = LazyAttribute(
        lambda x: max(fake.random_number(digits=5) / 100.0,0)#, 10.00)  # Desconto mínimo de 10.00
    )

class TransitTimeFactory(Factory):
    class Meta:
        model = TransitTime

    load_origin = LazyAttribute(lambda x: fake.city())
    load_destination = LazyAttribute(lambda x: fake.city())
    loaded_time = LazyAttribute(
        lambda x: max(fake.random_number(digits=4) / 100.0,10)#, 10.00)  # Desconto mínimo de 10.00
    )
    empty_time = LazyAttribute(
        lambda x: max(fake.random_number(digits=4) / 100.0,10)#, 10.00)  # Desconto mínimo de 10.00
    )


def create_transit_times():
    transit_times = [TransitTimeFactory().to_dict() for _ in range(5)]
    with open('transit_times.json', 'w') as f:
        json.dump(transit_times, f, indent=2, ensure_ascii=False)


@fixture
def create_model(
        simple_stock_node_factory,
        simple_clock,
        simple_product,
        simple_train
):
    def create(
            sim,
            n_trains,
            train_size = 6e3,
            process_times = [5,6,10,7,15,8.9,30,18,4,6]
    ):
        with open('tests/artifacts/transit_times.json', 'r') as f:
            data = json.load(f)

        load_points = []
        unload_points = []
        transit_times = []
        demands = []
        p_times = iter(process_times)
        for tt in data:
            d = simple_stock_node_factory(
                tt['load_destination'],
                clock=simple_clock,
                process='unload',
                has_replanisher=False,
                process_rate=train_size/next(p_times),
                initial_stock=randint(0,int(60e3))
            )
            unload_points.append(d)
            o = simple_stock_node_factory(
                tt['load_origin'],
                clock=simple_clock,
                process='load',
                has_replanisher=True,
                process_rate=train_size / next(p_times),
                initial_stock=randint(0,int(2*train_size))
            )
            load_points.append(o)
            transit_time = TransitTime(**tt)
            transit_times.append(transit_time)
            demand = Demand(
                flow=Flow(
                    origin=tt['load_origin'],
                    destination=tt['load_destination'],
                    product=simple_product
                ),
                volume=train_size*10
            )
            demands.append(demand)
        mesh = RailroadMesh(
            load_points=tuple(load_points),
            unload_points=tuple(unload_points),
            transit_times=transit_times
        )

        trains = [
            simple_train(
                simple_product,
                simple_clock,
                train_size=train_size,
                simulator=sim
            )

            for _ in range(n_trains)
        ]
        model = Railroad(
            mesh=mesh,
            trains=trains,
            demands=demands,
        )

        return model
    return create


@fixture
def create_simple_model(
        simple_stock_node_factory,
        simple_clock,
        simple_product,
        simple_train
):
    def create(
            sim,
            n_trains,
            train_size = 6e3,
            process_times = [10,7,15,7]
    ):
        o1 = "Souza"
        o2 = "Mendes"
        d = "Santos"
        data = [
          {
            "load_origin": o1,
            "load_destination": d,
            "loaded_time": 21.33,
            "empty_time": 10.0
          },
          {
            "load_origin": o2,
            "load_destination": d,
            "loaded_time": 68.51,
            "empty_time": 99.44
          }
        ]
        demands = [
            Demand(
                flow=Flow(
                    origin=o1,
                    destination=d,
                    product=simple_product
                ),
                volume=train_size * 10
            ),
            Demand(
                flow=Flow(
                    origin=o2,
                    destination=d,
                    product=simple_product
                ),
                volume=train_size * 3
            ),
        ]
        d = simple_stock_node_factory(
            d,
            clock=simple_clock,
            process='unload',
            has_replanisher=False,
            process_rate=train_size / 24,
            initial_stock=randint(0, int(60e3))
        )
        unload_points = [d]
        o1 = simple_stock_node_factory(
            o1,
            clock=simple_clock,
            process='load',
            has_replanisher=True,
            process_rate=train_size / 12,
            initial_stock=5*train_size
        )
        o2 = simple_stock_node_factory(
            o2,
            clock=simple_clock,
            process='load',
            has_replanisher=True,
            process_rate=train_size / 12,
            initial_stock=5*train_size
        )
        load_points = [o1, o2]


        transit_times = []

        for tt in data:

            transit_time = TransitTime(**tt)
            transit_times.append(transit_time)

        mesh = RailroadMesh(
            load_points=tuple(load_points),
            unload_points=tuple(unload_points),
            transit_times=transit_times
        )

        trains = [
            simple_train(
                simple_product,
                simple_clock,
                train_size=train_size,
                simulator=sim
            )

            for _ in range(n_trains)
        ]
        model = Railroad(
            mesh=mesh,
            trains=trains,
            demands=demands,
        )

        return model
    return create
