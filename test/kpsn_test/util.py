import contextlib
import os.path
import joblib

class PotentiallyLoadedData():
    def __init__(self,
        exists: bool,
        pth: str, 
        data: dict = None,
        verbose = False,
        nowrite: bool = False):
        
        self._exists = exists
        self._pth = pth
        self._data = data
        self._verbose = verbose
        self._nowrite = nowrite
    
    def exists(self): return self._exists
    
    def get(self, *keys):
        return (self._data[k] for k in keys)

    def save(self, **kwargs):
        if self._nowrite:
            if self._verbose:
                print("No-write (safe) mode prevented saving.")
            return
        if self._verbose:
            print(f"Saving to {self._pth}")
        joblib.dump(kwargs, self._pth)


@contextlib.contextmanager
def load_or_fit(pth, force_refit = False, verbose = True, safe = False):
    if not force_refit and os.path.exists(pth) and pth.endswith('.jl'):
        if verbose:
            print(f"Loading from {pth}")
        yield PotentiallyLoadedData(True, pth, joblib.load(pth), verbose, safe)
    else:
        yield PotentiallyLoadedData(False, pth, verbose = verbose, nowrite = safe)
    