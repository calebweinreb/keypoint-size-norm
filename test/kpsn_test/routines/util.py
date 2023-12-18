import importlib.util
import joblib as jl
import os.path
import time
from ruamel.yaml import YAML
import pathlib

def _read_yml(path):
    return YAML().load(pathlib.Path(path.strip()))

def find_file(key, fmt, override_paths = None, fallback_fmt = None):
    if override_paths is not None and key in override_paths:
        return override_paths[key]
    putative = fmt.format(key)
    fallback = fallback_fmt.format(key) if fallback_fmt is not None else None
    if os.path.exists(putative): return putative
    elif fallback is not None and os.path.exists(fallback): return fallback
    else: return putative

def save_results(output_fmt, fmt, ext, savefunc, omit = (), verbose = True):
    if fmt in omit: return
    path = output_fmt.format(name = fmt, ext = ext)
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
    elif isinstance(string, dict):
        return deepen_dict(string)
    elif string.endswith('.yml'):
        return _read_yml(string)
    else:
        def try_eval(s):
            try: return eval(s)
            except: return s
        ret = {}
        for kv in string.split(':'):
            kv = [s.strip() for s in kv.split('=')]
            if len(kv) == 1:
                kv = kv[0]
                if kv == '':
                    continue
                elif kv.endswith('.yml'):
                    nested_kvs = _read_yml(kv)
                    nested_kvs = deepen_dict(nested_kvs)
                    ret = update(ret, nested_kvs, warn_not_present=False, add=True)
                else:
                    deep_kv = deepen_dict({kv: True})
                    ret = update(ret, deep_kv, warn_not_present=False, add=True)
            else:
                deep_kv = deepen_dict({kv[0]: try_eval(kv[1])})
                ret = update(ret, deep_kv, warn_not_present=False, add=True)
        return ret
    

def load_cfg_list(cfgs):
    ret = {}
    for cfg in cfgs:
        ret = update(ret, load_cfg(cfg), warn_not_present=False, add=True)
    return ret

def load_cfg_lists(cfg_dict):
    return {k: load_cfg_list(v) for k, v in cfg_dict.items()}


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

    
def update(default, new, warn_not_present = True, add = False):
    """Overrwrite defaults given by the routine file for its configs"""
    def update_recurse(default, new, path):
        if isinstance(default, dict):
            # warn if there are instances in `new` that were not in `default`
            for k in new:
                if k not in default and warn_not_present:
                    print(f"Warning: tried to update {'.'.join(path + (k,))}, which is not in defaults.")
            # construct dictionary with values in default updated
            ret = {k: update_recurse(default[k], new[k], path = path + (k,))
                    if k in new else default[k]
                    for k in default}
            # if we are updating with addition, then any subtrees in new that
            # were not in `default` can be copied over
            if add:
                for k in new:
                    if k not in default:
                        ret[k] = new[k]
            return ret
        else:
            return new 
    return update_recurse(default, new, ())


def sort_cfg_list(args, shorthands = {}, base = None):
    """
    In: "d@key=value:key1=value1#kevals.yml"
    kevals.yml = {d: {key2: value2}}
    Out: dict(d = ["key=value:key1=value1", {key2: value2}])
    
    Or if shorthands = dict(d = data) with keyvals.yml still {d: ...}
    In: "data#key=value:key1=value1" "kevals.yml"
    Out: dict(data = ["key=value:key1=value1", {key2: value2}])
    """
    sorted_pairs = []
    for cfg_strs in args:
        for cfg_str in cfg_strs.split('#'):
            cfg_str = cfg_str.strip()
            
            if cfg_str.endswith('.yml'):
                cfg_data = _read_yml(cfg_str)
                for cfg_for, cfg_value in cfg_data.items():
                    if cfg_for in shorthands:
                        cfg_for = shorthands[cfg_for]
                    sorted_pairs.append((cfg_for, cfg_value))
                    
            else:
                split_str = cfg_str.split('@')
                if len(split_str) != 2:
                    print(f"WARNING: Could not determine destination for config {cfg_str}")
                    continue
                cfg_for = split_str[0].strip()
                if cfg_for in shorthands:
                    cfg_for = shorthands[cfg_for]
                cfg_value = split_str[1]
                sorted_pairs.append((cfg_for, cfg_value))

    if base is None: sorted_cfgs = {}
    else: sorted_cfgs = base

    for cfg_for, cfg_value in sorted_pairs:
        if cfg_for not in sorted_cfgs:
            sorted_cfgs[cfg_for] = []
        sorted_cfgs[cfg_for].append(cfg_value)
    
    return sorted_cfgs
        
            
                    
