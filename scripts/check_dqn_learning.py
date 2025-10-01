from models.TFRState import FlowState, ConstraintState, TrainState, TFRState
from models.demand import Flow
from models.states import ActivityState
from models.action_space import ActionSpace
from models.tfr_state_factory import TFRStateSpaceFactory
import dill
import torch
import pandas as pd


def str_to_state(text):
    sub_states = text.split('\n')
    flow_states = []
    train_states = []
    for ss in sub_states:
        if 'Fluxo' in ss:
            infos = ss.split(' ')
            od = infos[1]
            o, d = od.split('-')
            missing_volume = 1-float(infos[3][:-1])/100
            fs = FlowState(
                flow=Flow(origin=o, destination=d, product='product'),
                has_demand=missing_volume>0,
                missing_volume=missing_volume,
            )
            flow_states.append(fs)
        if 'Trem' in ss:
            infos = ss.split(' ')
            name = infos[1]
            local_activity = ' '.join(infos[3:])
            words = local_activity.split('-')
            activity = words[-1]
            local = '-'.join(words[:-1])
            is_initial = 'origin' in local
            ts = TrainState(
                name=name,
                local=local,
                activity=ActivityState(activity),                    
            )
            train_states.append(ts)
    contraint_states = [
        ConstraintState(name='xpto', is_blocked=False),
        ConstraintState(name='xpto', is_blocked=False),
        ConstraintState(name='xpto', is_blocked=False),
    ]
    state = TFRState(
        train_states=train_states,
        flow_states=flow_states,
        constraint_states=contraint_states,
        is_initial=is_initial
    )
    return state

base_model = 'tests/artifacts/simple_model_to_train_1_sim_v2.dill'

with open(base_model, 'rb') as f:
    model = dill.load(f)
state_space = TFRStateSpaceFactory(model)
action_space = ActionSpace(model.demands)
with open('serialized_models/dqn/policy_net_BALANCE_parallel_simple_model_15x4_TFRState_v2.dill', 'rb') as f:
    policy_net = dill.load(f)

with open('serialized_models/q_tables/q_table_10_nos_15_trens_balance.dill', 'rb') as f:
    q_table = dill.load(f)
report = []
for state_str in q_table:
    state = str_to_state(state_str)
    state_arr = state_space.to_array(state)
    state_tensor = torch.FloatTensor(state_arr).unsqueeze(0)
    demand_index = policy_net(state_tensor).argmax().item()
    best_action_by_dqn = action_space.demands[demand_index]
    best_q = 0
    best_action_by_q_table = None
    for action, q in q_table.get(state_str, {}).items():
        if q >= best_q:
            best_q = q
            best_action_by_q_table = action
    if best_action_by_q_table and not isinstance(best_action_by_q_table, str):
        best_action_by_q_table = [d for d in action_space.actions if d.flow==best_action_by_q_table][0]
    register = {
        "state": state_str,
        "Q_TABLE": best_action_by_q_table,
        "DQN": best_action_by_dqn,
        "consensus": best_action_by_dqn == best_action_by_q_table
    }
    report.append(register)

report = pd.DataFrame(report)
...
    
    
        