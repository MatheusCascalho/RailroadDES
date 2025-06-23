import dill

# with open('../serialized_models/chained_decision_map_trained_v1 (Cópia).dill', 'rb') as f:
with open('../tests/chained_decision_map_trained_v1_T0.dill', 'rb') as f:
    chained_decision_maps = dill.load(f)

transition = list(chained_decision_maps[0].transitions.values())[-1]

origin_ferrovia = transition.origin.name[0]
origin_demanda = transition.origin.name[1]
destination_demanda = transition.destination.name
evento = transition.name
fluxo_anterior = transition.origin.name[2]

print(f'Estado da ferrovia: \n{origin_ferrovia}')
print("="*20)

print(f'Estado da demanda origem: \n{origin_demanda}')
print("="*20)

print(f'Ação anterior: \n{fluxo_anterior}')
print("="*20)

print(f'Ação: \n{evento}')
print("="*20)

print(f'Fila: \n{evento.penalty()}')
print("="*20)

print(f'Recompensa: \n{evento.reward()}')
print("="*20)

print(f'Estado da demanda destino: \n{destination_demanda}')
print("="*20)