import jax.tree_util as pt
import jax.numpy as jnp

class ReportTrace():
    def __init__(self, n_steps):
        self._n_steps = n_steps 
        self._tree = None

    def initialize(self, report):
        self._tree = pt.tree_map(
            lambda report_leaf: jnp.zeros(
                (self._n_steps,) + report_leaf.shape,
                report_leaf.dtype),
            report)

    def record(self, reports, step):
        if self._tree is None: self.initialize(reports)
        self._tree = pt.tree_map_with_path(
            lambda pth, trace, report_leaf:
            trace.at[step].set(report_leaf)
            if report_leaf.shape == trace[step].shape
            else exec('raise ValueError("Report at path ' + 
                     f'{pth} had shape {report_leaf.shape} when '
                      'trace was initialized with shape' + 
                     f'{trace[step].shape}")'),
            self._tree, reports
        )

    def read(self): return self._tree

    def n_leaves(self):
        return len(pt.tree_flatten(self._tree)[0])
    
    def plot(self, axes, label_mode = 'title', **artist_kws):
        zipped_paths_leafs, _ = pt.tree_flatten_with_path(self._tree)
        for ax, (path, leaf) in zip(axes, zipped_paths_leafs):
            plottable = leaf.reshape([len(leaf), -1])
            ax.plot(plottable, **artist_kws)
            if label_mode == 'title': ax.set_title(_keystr(path))
            elif label_mode == 'yaxis': ax.set_ylabel(_keystr(path))
            else: ax.set_xlabel(_keystr(path))


def _single_key_repr(tree_key):
    if isinstance(tree_key, pt.DictKey): return tree_key.key
    if isinstance(tree_key, pt.SequenceKey): return tree_key.idx
    if isinstance(tree_key, pt.GetAttrKey): return tree_key.name
def _keystr(path):
    return "/".join(_single_key_repr(k) for k in path)
            



        
