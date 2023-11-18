import importlib.util
import joblib as jl
import os.path
import time
import json


def find_file(key, fmt, override_paths = None, fallback_fmt = None):
    if override_paths is not None and key in override_paths:
        return override_paths[key]
    putative = fmt.format(key)
    fallback = fallback_fmt.format(key) if fallback_fmt is not None else None
    if os.path.exists(putative): return putative
    elif fallback is not None and os.path.exists(fallback): return fallback
    else: return putative

def save_results(output_fmt, fmt, savefunc, omit = (), verbose = True):
    if fmt in omit: return
    path = output_fmt.format(fmt)
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        if verbose:
            print(f"Creating directory {dirname}")
        os.makedirs(dirname)
    savefunc(path)
    if verbose:
        print(f"Saved outputs: {fmt}")


def load_routine(filename, root = None):
    """
    Load the global variable `model` from filename, the implication being that
    the object is a `network_manager.LayerMod` to use as an attention model.
    """
    # Run the model file
    try:
        routine = importlib.import_module(
            f'{root}.{filename}' if root is not None else filename)
    except ImportError:
        time_hash = hash(time.time())
        fname_base = os.path.basename(filename)
        spec = importlib.util.spec_from_file_location(
            f"routine_{fname_base}_{time_hash}", filename)
        routine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(routine)
    return routine


def load_cfg(string):
    """
    Either load the JSON file pointed to by `string` or parse it as configs
    (`:`-sepearated statements of KEY=VALUE, where `VALUE` is parsed as a
    Python expression or string if evaluation fails, for example
    `"layer=(0,1,0):beta=4.0"` would become {'layer':(0,4,0), 'beta': 4.0})
    """
    if string == '-':
        return {}
    if string is None:
        return {}
    elif string.endswith('.json'):
        with open(string) as f:
            return json.load(f)
    else:
        def try_eval(s):
            try: return eval(s)
            except: return s
        
        keyvals = [[kv.strip() for kv in s.split('=')]
                   for s in string.split(':')]
        keyvals = [
            (kv[0], True) if len(kv) == 1 else (kv[0], kv[1])
            for kv in keyvals if not ''.join(kv) == '']
        
        return deepen_dict({
                try_eval(k): # LHS = key
                try_eval(v)  # RHS = val
            for k, v in keyvals
        })


def deepen_dict(d):
    """{'a.b': 'c'} -> {'a': {'b': 'c'}}"""
    ret = {}
    for k in d:
        path = k.split('.')
        curr_dir = ret 
        while len(path) > 1:
            to_enter = path[0]
            if to_enter not in curr_dir: curr_dir[to_enter] = {}
            curr_dir = curr_dir[to_enter]
            path = path[1:]
        curr_dir[path[0]] = d[k]
    return ret

    
def update(default, new, warn_not_present = True):
    """Overrwrite defaults given by the routine file for its configs"""
    def update_recurse(default, new, path):
        if isinstance(default, dict):
            for k in new:
                if k not in default and warn_not_present:
                    print(f"Warning: tried to update {'.'.join(path + (k,))}, which is not in defaults.")
            return {k: update_recurse(default[k], new[k], path = path + (k,))
                    if k in new else default[k]
                    for k in default}
        else:
            return new 
    return update_recurse(default, new, ())
