import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from kpsn_test import visualize as viz
from kpsn.util import skeleton, alignment, keypt_io
from kpsn_test.clustering import apply_clusters, masks_and_logits

def plot(
    plot_name,
    dataset,
    cfg,
    **kwargs
    ):

    bhv_counts = dataset['metadata'][cfg['counts']]
    sess_bhv_tags = dataset['metadata'][cfg['bhv']]
    ex_sess = list(bhv_counts.keys())[0]
    N = len(bhv_counts)
    L = bhv_counts[ex_sess].size
    clusterpal = sns.color_palette('Set2', n_colors = L)

    # get true counts of behavior matching each cluster
    # kmeans = dataset['metadata']['shared']['kmeans']
    # kpts = {sess: 
    #     dataset['keypts'][dataset['metadata']['session_slice'][sess]].astype(np.float32)
    #     for sess in dataset['metadata']['session_slice']}
    # true_counts = {sess: masks_and_logits(
    #     kmeans.transform(kpts[sess]).argmin(axis = 1),
    #     L)[1]
    #     for sess in dataset['metadata']['session_slice']}

    # find example sessions with the correct 'bhv' tag
    sessions = [dataset['metadata']['session_ix'].inverse[i] for i in range(N)]
    tag_list = [sess_bhv_tags[sess] for sess in sessions]
    bhvs, ex_ixs = np.unique(tag_list, return_index=True)
    ex_sessions = [sessions[i] for i in ex_ixs]

    fig, ax = plt.subplots(1, L, figsize = (1.5 * L, 2), sharex = True, sharey = True)
    for i_bhv, (bhv, ex_sess) in enumerate(zip(bhvs, ex_sessions)):
        ax[i_bhv].bar(np.arange(L), bhv_counts[ex_sess], color = clusterpal)
        # ax[i_sess].plot(np.arange(L), true_counts[sess], 'ko')
        ax[i_bhv].set_title(bhv)
        sns.despine(ax = ax[i_bhv])
    ax[0].set_ylabel("Counts")
    ax[0].set_xlabel("Cluster")
    fig.tight_layout()

    return {plot_name: fig}

defaults = dict(
    counts = 'bhv_counts',
    bhv = 'bhv'
)