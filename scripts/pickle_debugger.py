import pickle
import types

def is_picklable(obj):
    try:
        pickle.dumps(obj)
        return True
    except Exception as e:
        return False

def find_pickle_issues(obj, path='obj', visited=None, max_depth=20):
    if visited is None:
        visited = set()

    results = []
    obj_id = id(obj)
    if obj_id in visited or max_depth <= 0:
        return results
    visited.add(obj_id)

    # Check if the object itself is not picklable
    try:
        pickle.dumps(obj)
    except Exception as e:
        results.append((path, type(obj), str(e)))
        # Try to go deeper if possible
        if isinstance(obj, dict):
            for k, v in obj.items():
                results += find_pickle_issues(v, f'{path}[{repr(k)}]', visited, max_depth-1)
        elif hasattr(obj, '__dict__'):
            for attr_name, attr_val in vars(obj).items():
                results += find_pickle_issues(attr_val, f'{path}.{attr_name}', visited, max_depth-1)
        elif isinstance(obj, (list, tuple, set)):
            for i, item in enumerate(obj):
                results += find_pickle_issues(item, f'{path}[{i}]', visited, max_depth-1)
    return results
