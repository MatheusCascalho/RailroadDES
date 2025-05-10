from models.railroad_mesh import TransitTime, RailroadMesh
from models.des_model import Railroad
from faker import Faker
from factory import Factory, LazyAttribute
import json
from pytest import fixture
from models.demand import Demand, Flow
from models.router import RandomRouter

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
            train_size = 6e3
    ):
        with open('artifacts/transit_times.json', 'r') as f:
            data = json.load(f)

        load_points = []
        unload_points = []
        transit_times = []
        demands = []
        for tt in data:
            d = simple_stock_node_factory(
                tt['load_destination'],
                clock=simple_clock,
                process='unload',
                has_replanisher=False
            )
            unload_points.append(d)
            o = simple_stock_node_factory(
                tt['load_origin'],
                clock=simple_clock,
                process='load',
                has_replanisher=True
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
        router = RandomRouter(demands=demands)
        model = Railroad(
            mesh=mesh,
            trains=trains,
            demands=demands,
            router=router
        )

        return model
    return create
