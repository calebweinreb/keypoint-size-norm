import joblib, os, glob, re
from bidict import bidict
import numpy as np

def modata_name_func(path):
    m = re.search(r"(?:/.*)+/\d{2}_\d{2}_\d{2}_(\d+wk_m\d+)\.npy",
              path)
    if m is None:
        raise ValueError(f"using modata_name_func on {path}")
    return m.group(1)

keypt_names_raw = np.array([
    'shldr',
    'back',
    'hips',
    't_base',
    't_tip',
    'head',
    'l_ear',
    'r_ear',
    'nose',
    'lr_knee',
    'lr_foot',
    'rr_knee',
    'rr_foot',
    'lf_foot',
    'rf_foot'
])
# keypoint naming/finding tools for raw array form
raw_keypt_by_name = {n: i for i, n in enumerate(keypt_names_raw)}
keypt_parents_raw = np.array([0,0,1,2,3,0,5,5,5,2,9,2,11,0,0])
keypt_parent_names_raw = keypt_names_raw[keypt_parents_raw]
# keypoint naming/finding tools for *processed* array form
keypt_names = np.delete(keypt_names_raw, raw_keypt_by_name['t_tip'])
keypt_by_name = {n: i for i, n in enumerate(keypt_names)}
keypt_parent_names = np.array([
    keypt_parent_names_raw[raw_keypt_by_name[n]]
    for n in keypt_names])
keypt_parents = np.array([keypt_by_name[n] for n in keypt_parent_names])



def from_gimbal(path):
    """
    Load a raw keypoint array from a gimbal output file.
    """
    from_raw_array(joblib.load(path)['positions_medfilter'])

def from_npy_formatted_path(path):
    # From .../{date}_{age}wk_m{id}.npy
    metadata_segments = os.path.basename(path).split('.')[0].split('_')
    metadata = dict(
        date = '_'.join(metadata_segments[:3]),
        age = int(metadata_segments[3].rstrip('wk')),
        id = int(metadata_segments[4].lstrip('m'))
    )
    keypoints = from_raw_array(np.load(path))
    return metadata, keypoints

def from_raw_array(raw_arr):
    """
    Clean up a raw keypoint array as stored on disk.

    Loaded from a npy or gimbal file, arr has shape (t, keypt, dim).
    This function is a space for general data cleaning operations, but
    for now all we do is drop the tail-tip keypoint.

    Parameters
    ----------
    raw_arr: array-like, (t, keypt, dim)
        Raw keypoint array to be cleaned
    
    Returns
    -------
    cleaned: array-like, (t, keypt, dim)
        Cleaned data, possibly with a new shape.
    """
    no_tail = np.delete(raw_arr, raw_keypt_by_name['t_tip'], axis = 1)
    return no_tail



def npy_dataset(dirname, whitelist = None, name_func = modata_name_func):
    """
    Load a set of .npy keypoint data.

    Parameters
    ----------
    dirname: str
        Directory in which to search for data files.
    
    whitelist: str, Iterable[str], or None
        If an iterable of strings, then should contain the filenames
        to be loaded from `dirname`. If a string, then will be
        treated as a path to a file containing the filenames to be
        loaded from `dirname`, one per line. If `None`, then
        `dirname` will be searched for .npy files.

    Returns
    -------
    metadata: dict[str, List]
        Metadata as returned by `npy_file_dataset`
    
    keypts: List[array(t, keypt, dim)]
        Keypoint data.
    """
    glob_files = sorted(glob.glob(f'{dirname}/*.npy'))
    
    if whitelist is not None:
        if isinstance(whitelist, str):
            with open(whitelist, 'r') as f:
                files = [
                    os.path.normpath(os.path.join(dirname, l.strip()))
                    for l in f.readlines()]
        else:
            files = [
                os.path.normpath(os.path.join(dirname, f.strip()))
                for f in glob_files if name_func(f.strip()) in whitelist
            ]
    else:
        files = glob_files
    
    meta, keypts = npy_file_dataset(files)
    meta['file'] = files
    
    return meta, keypts

def npy_file_dataset(files):
    """
    Load dataset from a list of .npy keypoint data.

    Parameters
    ----------
    files: Iterable[str]
        Paths to .npy files containing keypoint data from recording
        sessions.
    
    Returns
    -------
    metadata: dict[str, List]
        Metadata under keys `'date'` (of session), `'age'` (of
        subject at time of session) and `'id'` (of the subject).
        Each entry is a list with length matching that of `keypts`.
    
    keypts: List[array(t, keypt, dim)]
        List of keypoint data arrays, one per session.
    """
    metadata = {key: [] for key in ['date', 'age', 'id']}
    keypoints = []
    for fname in files:
        m, k = from_npy_formatted_path(fname)
        for key in metadata: metadata[key].append(m[key])
        keypoints.append(k)
    return metadata, keypoints

def test_dataset(dirname = "data"):
    metadata, keypts = npy_dataset(dirname)
    metadata, keypts = select_subset(metadata, keypts, [1, 39, 0, 23, 22, 2])
    metadata, keypts = split_videos(metadata, keypts, 3)
    keypts = subsample_time(keypts, 5)
    return metadata, keypts

def truncate_videos(keypts, n_frames):
    return [k[:n_frames] for k in keypts]

def split_videos(metadata, keypts, n_parts, video_key = 'subj_vid', new_id = 'split_id', src_id = 'id'):
    splitten = [np.array_split(k, n_parts) for k in keypts]
    new_kpts = [splitten[i][j] for i in range(len(keypts)) for j in range(n_parts) ]
    new_metadata = {
        k: [val for val in v for _ in range(n_parts)]
        for k, v in metadata.items()
    }
    new_metadata[src_id] = [
        metadata["id"][j] for j in range(len(keypts)) for i in range(n_parts)]
    new_metadata[new_id] = [
        f'{metadata["id"][j]}.{i}' for j in range(len(keypts)) for i in range(n_parts)]
    new_metadata[video_key] = [
        i for _ in range(len(keypts)) for i in range(n_parts)]
    return new_metadata, new_kpts

def split_videos_dict(
    metadata, keypts, n_parts,
    video_key = 'subj_vid',
    src_key = 'presplit',
    name_fmt = lambda sess, i: f'{sess}.{i}'
    ):

    split_kpts = {}
    new_metadata = {video_key: {}, src_key: {}, **{k: {} for k in metadata}}
    for src_sess in keypts:

        split_arrs = np.array_split(keypts[src_sess], n_parts)
        for i, arr in enumerate(split_arrs):
            sess_name = name_fmt(src_sess, i)
            split_kpts[sess_name] = arr
        
            for k in metadata:
                new_metadata[k][sess_name] = metadata[k][src_sess]
            new_metadata[video_key][sess_name] = i
            new_metadata[src_key][sess_name] = src_sess

    return new_metadata, split_kpts

def select_subset(metadata, keypts, ixs):
    new_kpts = [keypts[i] for i in ixs]
    new_metadata = {k: [v[i] for i in ixs] for k, v in metadata.items()}
    return new_metadata, new_kpts


def get_flat_subset(session_names, slices, keypts):
    new_feats = {s: keypts[slices[s]] for s in session_names}
    new_slices, all_new_feats = to_flat_array(new_feats)
    new_sess_ix, new_sess_ids = ids_from_slices(all_new_feats, new_slices)
    return new_feats, all_new_feats, new_slices, new_sess_ix, new_sess_ids

def subsample_time(kpts, factor):
    return [k[::factor] for k in kpts]

def get_groups(metadata, key):
    group_keys, group_inv = np.unique(metadata[key], return_inverse = True, axis = 0)
    group_ixs = [np.where(group_inv == i)[0] for i in range(group_inv.max() + 1)]
    metadata[key] = np.array(metadata[key])
    return group_keys, group_ixs

def get_groups_dict(metadata_val):
    """
    Parameters
    ----------
    metadata_val : dict
        Mapping session names to metadata values
    """
    groups = {}
    for sess_name, val in metadata_val.items():
        if val not in groups: groups[val] = []
        groups[val].append(sess_name)
    sorted_keys = sorted(groups.keys())
    return sorted_keys, tuple(groups[k] for k in sorted_keys)

def metadata_zip(*vals):
    return {k: tuple(v[k] for v in vals) for k in vals[0]}


def to_feats(kpts):
    return kpts.reshape(kpts.shape[:-2] + (kpts.shape[-2] * kpts.shape[-1],))

def from_feats(kpts, n_dim):
    return kpts.reshape(kpts.shape[:-1] + (-1, n_dim))

def to_flat_array(feature_dict):
    """
    Merge a dictionary of arrays.

    features: dict[any, array[n_samp, *feature_dims]]
    Returns
    -------
    slices: dict[any, slice]
        Slices to recover original arrays from the merged array.
    flat_array: (sum(n_samp), *feature_dims)
        Merged array.
    """
    i = 0
    to_concat = []
    slices = {}
    for k in feature_dict:
        slices[k] = slice(i, i + feature_dict[k].shape[0])
        to_concat.append(feature_dict[k])
        i += feature_dict[k].shape[0]
    return slices, np.concatenate(to_concat)

def groups_to_flat_arrays(group_keys, group_ixs, features):
    """
    Merge recording sessions by group.

    Parameters
    ----------
    group_keys, group_ixs: as from get_groups()
    features: list[array[n_samp, *feature_dims]]

    Returns
    -------
    slices: dict[group_key, dict[any, slice]]
        Mapping from groups to slices that recover original videos
        from the flattened arrays.
    
    keypt_sets: dict[group_key, array(merged_n_samp, *feature_dims)]
        Mapping from groups to merged keypoint arrays.
    """
    slices, keypt_sets = {}, {}
    for group_key, group in zip(group_keys, group_ixs):
        s, k = to_flat_array({vid: features[vid] for vid in group})
        slices[group_key] = s
        keypt_sets[group_key] = k.reshape([k.shape[0], -1])
    return slices, keypt_sets

def ids_from_slices(flat_dataset, slices):
    """
    Parameters
    ----------
    flat_dataset : (n_samp, *feature_dims)
        Flattened dataset with total number of samples along first dimension.
    slices : Dict[Any, slice]
        Indexing slices returned by to_flat_array
    
    Returns
    -------
    id_by_name : bidict[Any, int]
        Bidirectional mapping between session names and session ids
    sample_ids : np.ndarray[int], (n_samp,)
        Sesssion id of each sample in the flat dataset
    """
    id_by_name = bidict()
    ids = np.full([len(flat_dataset)], -1)
    for i, (name, slc) in enumerate(slices.items()):
        id_by_name[name] = i
        ids[slc] = i
    assert (ids == -1).sum() == 0, ("Slices do not tile full dataset")

    return id_by_name, ids

def apply_across_flat_array(f, slices, *arrays):
    """
    f: (vid, *arrays) => any
    slices: dict[vid: slice or array[int]]
    Returns:
    result: dict[vid: any]
    """
    return {
        vid: f(vid, *(arr[slices[vid]] for arr in arrays))
        for vid in slices}

