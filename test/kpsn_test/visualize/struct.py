import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from kpsn.util import keypt_io

def axes_by_age_and_id(
    subject_ages,
    subject_ids,
    summative_col = False,
    axes_off = True,
    sharex = True,
    sharey = True,
    figsize = None):
    
    if isinstance(subject_ages, dict):
        ages, age_groups = keypt_io.get_groups_dict(subject_ages)
        ageids, ageid_groups = keypt_io.get_groups_dict(
            keypt_io.metadata_zip(subject_ages, subject_ids))
    else:
        ages, age_groups = keypt_io.get_groups(
            dict(by = subject_ages), 'by')
        ageids, ageid_groups = keypt_io.get_groups(
            dict(by = list(zip(subject_ages, subject_ids))), 'by')
    
    max_per_age = max(len(grp) for grp in age_groups)

    summ_col_ofs = 1 if summative_col else 0
    if figsize is not None:
        figsize = (figsize[0] * (max_per_age + 1), figsize[1] * len(ages))
    fig, ax = plt.subplots(len(ages), max_per_age + summ_col_ofs, sharex = sharex, sharey = sharey, figsize = figsize)

    if axes_off:
        for i_age in range(len(ages)):
            for i_col in range(max_per_age + summ_col_ofs):
                ax[i_age, i_col].set_xticks([])
                ax[i_age, i_col].set_yticks([])
                sns.despine(ax = ax[i_age, i_col], bottom = True, left = True)

    if summative_col:
        summative_col_it = lambda: [(ax[i, 0], age, age_groups[i]) for i, age in enumerate(ages)]

    def ax_iter():
        age_rows = dict(zip(sorted(ages), list(range(len(ages)))))
        col_counter = {age: 0 for age in ages}
        for i, (ageid, ageid_group) in enumerate(zip(ageids, ageid_groups)):
            curr_ax = ax[age_rows[ageid[0]], col_counter[ageid[0]] + summ_col_ofs]
            col_counter[ageid[0]] += 1
            for vid_id in ageid_group:
                if summative_col:
                    summ_ax = ax[age_rows[ageid[0]], 0]
                    yield curr_ax, ageid[0], ageid[1], vid_id, summ_ax
                else:
                    yield curr_ax, ageid[0], ageid[1], vid_id

    ret = fig, ax, ax_iter
    if summative_col: ret = ret + (summative_col_it,)
    return ret


def flat_grid(total, n_col, ax_size, subplot_kw = {}, return_grid = False):
    n_row = int(np.ceil(total / n_col))
    fig, ax = plt.subplots(
        n_row, n_col,
        figsize = (ax_size[0] * n_col, ax_size[1] * n_row),
        **subplot_kw)
    ax = np.array(ax)
    if ax.ndim == 1: ax = ax[None, :]
    elif ax.ndim == 0: ax = ax[None, None]
    ax_ravel = ax.ravel()
    for a in ax_ravel[total:]:
        a.set_axis_off()
    
    ret = fig, ax_ravel[:total]
    if return_grid: ret += (ax,)
    return ret


# def axis_color(ax, color):
#     for side in ['top', 'bottom', 'left', 'right']:
#         ax.spines[side].set_color(color)
#     ax.xaxis.label.set_color(color)
#     ax.yaxis.label.set_color(color)
#     ax.tick_params(axis = 'x', colors = color)
#     ax.tick_params(axis = 'y', colors = color)


