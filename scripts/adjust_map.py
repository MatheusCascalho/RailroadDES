import dill

with open('../tests/decision_map_trained_v0.dill', 'rb') as f:
    model = dill.load(f)

new_map = {}

for key, values in model.items():
    i = key.find('train_0')
    if key[:i] not in new_map:
        new_map[key[:i]] = values
    else:
        new_map[key[:i]].extend(values)

with open('../tests/decision_map_trained_v1.dill', 'wb') as f:
    dill.dump(new_map, f)

print(f"Estados reduzidos de {len(model)} para {len(new_map)}")
ape = {len(v) for v in new_map.values()}
print(f"Ações por estado: {ape}")
