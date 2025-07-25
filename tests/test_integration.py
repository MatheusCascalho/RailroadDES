import pytest
from datetime import timedelta

from models.DQNRouter import DQNRouter
from models.event import DecoratedEventFactory
from models.event_calendar import EventCalendar
from models.exceptions import TrainExceptions
from hypothesis import given, strategies as st
from hypothesis import HealthCheck, settings
from models.operated_volume import OperatedVolume
from models.railroad_mesh import TransitTime, RailroadMesh
# from tests.artifacts.train_artifacts import simple_train
from models.demand import Demand, Flow
from models.railroad import Railroad
from models.des_simulator import DESSimulator
from models.gantt import Gantt
from models.stock_graphic import StockGraphic
from models.system_evolution_memory import RailroadEvolutionMemory
from models.tfr_state_factory import TFRStateSpaceFactory
from models.target import SimpleTargetManager
from tests.artifacts.simulator_artifacts import FakeSimulator
from pytest import mark
import dill

@mark.skip(reason="Despriorização de testes de integração.")
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    steps=st.lists(st.integers(min_value=1,max_value=360), min_size=3, max_size=10)
)
def test_stock_node_should_block_train_in_queue_to_enter_when_stock_is_empty(
        simple_clock,
        simple_stock_node,
        simple_product,
        simple_train,
        simple_simulator,
        steps
):
    # Arrange
    train = simple_train(train_size=6e9)
    process_time = timedelta(hours=5)

    # Act
    sim = FakeSimulator(clock=simple_clock)
    train.arrive(node=simple_stock_node)
    simple_stock_node.receive(simulator=simple_simulator,train=train)

    for i in range(10):
        try:
            simple_stock_node.process(simulator=sim)
            train.start_load(simulator=sim, process_time=process_time)
            simple_clock.jump(process_time)
            train.finish_load(simulator=sim, node=simple_stock_node)
            simple_stock_node.maneuver_to_dispatch(simulator=sim)
        except TrainExceptions as e:
            simple_clock.jump(timedelta(hours=10))
            simple_stock_node.process(simulator=sim)
        assert not train._in_slot

@mark.skip(reason="Despriorização de testes de integração.")
def test_simulation(simple_train, simple_stock_node, simple_product, simple_clock):

    sim = FakeSimulator(clock=simple_clock)
    train = simple_train(product=simple_product, clk=simple_clock)
    process_time = timedelta(hours=5)

    # Simulacao
    train.arrive(simulator=sim, node=simple_stock_node)
    simple_stock_node.receive(train)
    simple_stock_node.process(simulator=sim)
    train.start_load(simulator=sim, process_time=process_time)
    simple_clock.jump(process_time)
    train.finish_load(simulator=sim, node=simple_stock_node)
    simple_stock_node.maneuver_to_dispatch(simulator=sim)
    assert True

@mark.skip(reason="Despriorização de testes de integração.")
def test_stock_node_simulation(simple_train, simple_stock_node, simple_clock):

    sim = FakeSimulator(clock=simple_clock)
    product = "product"
    train_size = 6e3
    train = simple_train(product, simple_clock, train_size=train_size, simulator=sim)

    node = simple_stock_node
    clk = simple_clock
    process_time = timedelta(hours=5)

    # Simulacao
    train.arrive(simulator=sim, node=node)
    node.receive(train)
    node.process(simulator=sim)
    try:
        train.start_load(simulator=sim, process_time=process_time)
    except TrainExceptions as e:

        ##### encapsular no EventExceptionHandler ########
        blocked_constraints = [c for c in node.process_constraints if c.is_blocked()]
        reasons = [
            c.reason(
                train_size=train.capacity,
                try_again=node.time_to_try_again(
                    product=train.product,
                    volume=train.capacity,
                    process=train.current_process_name
                )
            ) for c in blocked_constraints
        ]
        t = max([r.time_to_try_again for r in reasons])
        print(e)
        clk.jump(t)
        ##### encapsular no EventExceptionHandler ########
        node.process(simulator=sim)

        train.start_load(simulator=sim, process_time=process_time)
        clk.jump(timedelta(hours=3))
        node.process(simulator=sim)

        clk.jump(timedelta(hours=2))
        train.finish_load(simulator=sim, node=node)
        # node.maneuver_to_dispatch(simulator=sim)
        clk.jump(timedelta(hours=5))
        node.dispatch()


    assert True

@pytest.fixture
def simple_model(simple_stock_node_factory, simple_train, simple_clock):
    def create_model(
            sim,
            demand=[3500, 0],
            n_trains=1,
            terminal_times = [7],
            port_times=[6, 10],
            queue_capacity=50):
        load_points = (
            simple_stock_node_factory('origin', clock=simple_clock, process='load'),
        )

        unload_points = (
            simple_stock_node_factory('destination', clock=simple_clock, process='unload', has_replanisher=False),
        )

        transit_times = [
            TransitTime(
                load_origin=load_points[0].name,
                load_destination=unload_points[0].name,
                empty_time=timedelta(hours=17),
                loaded_time=timedelta(hours=20)
            ),
            # TransitTime(
            #     load_origin=load_points[0].name,
            #     load_destination=unload_points[1].name,
            #     empty_time=timedelta(hours=17),
            #     loaded_time=timedelta(hours=20)
            # ),
        ]

        mesh = RailroadMesh(
            load_points=load_points,
            unload_points=unload_points,
            transit_times=transit_times
        )
        product = "product"
        train_size = 6e3

        trains = [
            simple_train(
                product,
                simple_clock,
                train_size=train_size,
                simulator=sim
            )

            for _ in range(n_trains)
        ]

        demands = [
            Demand(
                flow=Flow('origin', 'destination', product),
                volume=demand[0],
            )
        ]

        model = Railroad(mesh=mesh, trains=trains, demands=demands)
        return model
    return create_model

@mark.skip(reason="Despriorização de testes de integração.")
def test_model(simple_model, simple_clock):
    sim = DESSimulator(clock=simple_clock)

    m = simple_model(sim=sim)
    sim.simulate(model=m, time_horizon=timedelta(days=10))
    ...
    # m.starting_events(simulator=sim)

@mark.skip(reason="Despriorização de testes de integração.")
def test_stock_based_model(create_model, simple_clock):
    sim = DESSimulator(clock=simple_clock)
    model = create_model(sim=sim, n_trains=15)
    # with open('artifacts/model_2.dill', 'wb') as f:
    #     dill.dump(model, f)
    sim.simulate(model=model, time_horizon=timedelta(days=20))
    decision_map = {s: [{'penalty': t.penalty().total_seconds()/(60*60), 'reward': t.reward()} for t in model.router.decision_map[s]] for s in model.router.decision_map}
    sg = StockGraphic(list(model.mesh.load_points) + list(model.mesh.unload_points))
    sg.get_figures()[0].show()
    Gantt().build_gantt_with_all_trains(model.trains)
    Gantt().build_gantt_by_trains(model.trains)
    op_vol = OperatedVolume(model.router.completed_tasks)
    op_vol.plot_operated_volume().show()
    print(op_vol.operated_volume_by_flow())

@mark.skip(reason="Despriorização de testes de integração.")
def test_simulation_with_snapshot(create_model, simple_clock):
    memory = RailroadEvolutionMemory()
    event_factory = DecoratedEventFactory(
        pre_method=memory.save_previous_state,
        pos_method=memory.save_consequence
    )
    calendar = EventCalendar(event_factory=event_factory)
    sim = DESSimulator(clock=simple_clock, calendar=calendar)
    model = create_model(sim=sim, n_trains=15)
    with open('artifacts/model_to_train_15.dill', 'wb') as f:
        dill.dump(model, f)
    state_space = TFRStateSpaceFactory(model)
    router = DQNRouter(state_space=state_space, demands=model.demands)

    model.router = router
    memory.railroad = model
    memory.add_observers([router])
    with router:
        sim.simulate(model=model, time_horizon=timedelta(days=60))

    from collections import Counter
    print(memory)
    print(Counter([s.reward for s in memory.memory]))

    # sg = StockGraphic(list(model.mesh.load_points) + list(model.mesh.unload_points))
    # sg.get_figures()[0].show()
    # Gantt().build_gantt_with_all_trains(model.trains, final_date=model.mesh.load_points[0].clock.current_time)
    # Gantt().build_gantt_by_trains(model.trains)
    op_vol = OperatedVolume(model.router.completed_tasks)
    op_vol.plot_operated_volume().show()
    print(op_vol.operated_volume_by_flow())


@mark.skip(reason="Despriorização de testes de integração.")
def test_simulation_with_dqn_and_target(create_model, simple_clock):
    memory = RailroadEvolutionMemory()
    event_factory = DecoratedEventFactory(
        pre_method=memory.save_previous_state,
        pos_method=memory.save_consequence
    )
    calendar = EventCalendar(event_factory=event_factory)
    sim = DESSimulator(clock=simple_clock, calendar=calendar)
    model = create_model(sim=sim, n_trains=15)
    target = SimpleTargetManager(demand=model.demands)
    with open('artifacts/model_to_train_15.dill', 'wb') as f:
        dill.dump(model, f)
    state_space = TFRStateSpaceFactory(model)
    router = DQNRouter(
        state_space=state_space,
        demands=model.demands,
        policy_net_path='../serialized_models/policy_net_150x6_TFRState_v1_TargetBased.dill',
        target_net_path='../serialized_models/target_net_150x6_TFRState_v1_TargetBased.dill',
        explortation_method=target.furthest_from_the_target
    )

    model.router = router
    memory.railroad = model
    memory.add_observers([router])
    with router:
        sim.simulate(model=model, time_horizon=timedelta(days=60))

    from collections import Counter
    print(memory)
    print(Counter([s.reward for s in memory.memory]))

    # sg = StockGraphic(list(model.mesh.load_points) + list(model.mesh.unload_points))
    # sg.get_figures()[0].show()
    # Gantt().build_gantt_with_all_trains(model.trains, final_date=model.mesh.load_points[0].clock.current_time)
    # Gantt().build_gantt_by_trains(model.trains)
    op_vol = OperatedVolume(model.router.completed_tasks)
    op_vol.plot_operated_volume().show()
    print(op_vol.operated_volume_by_flow())
