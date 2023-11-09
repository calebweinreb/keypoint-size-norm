import matplotlib.pyplot as plt
import seaborn as sns

from kpsn.util import keypt_io

def axes_by_age_and_id(
    subject_ages,
    subject_ids,
    summative_col = False,
    axes_off = True,
    sharex = True,
    sharey = True,
    figsize = None):
    
    ages, age_groups = keypt_io.get_groups(dict(by = subject_ages), 'by')
    ageids, ageid_groups = keypt_io.get_groups(dict(by = list(zip(subject_ages, subject_ids))), 'by')
    max_per_age = max(len(grp) for grp in age_groups)

    summ_col_ofs = 1 if summative_col else 0
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


def flat_grid(total, n_col):
    n_row = int(np.ceil(total / n_col))
    fig, ax = plt.subplots(n_row, n_col, figsize = (3 * n_col, 2 * n_row))
    return fig, ax


