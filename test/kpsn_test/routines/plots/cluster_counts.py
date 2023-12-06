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

    bhv_counts = dataset['metadata']['bhv_counts']
    ex_sess = list(bhv_counts.keys())[0]
    L = bhv_counts[ex_sess].size
    clusterpal = sns.color_palette('Set2', n_colors = L)

    kmeans = dataset['metadata']['shared']['kmeans']
    kpts = {sess: 
        dataset['keypts'][dataset['metadata']['session_slice'][sess]].astype(np.float32)
        for sess in dataset['metadata']['session_slice']}
    true_counts = {sess: masks_and_logits(
        kmeans.transform(kpts[sess]).argmin(axis = 1),
        L)[1]
        for sess in dataset['metadata']['session_slice']}

    fig, ax = plt.subplots(1, L, figsize = (1.5 * L, 2), sharex = True, sharey = True)
    for i_sess, sess in enumerate(bhv_counts):
        ax[i_sess].bar(np.arange(L), bhv_counts[sess], color = clusterpal)
        ax[i_sess].plot(np.arange(L), true_counts[sess], 'ko')
        ax[i_sess].set_title(sess)
        sns.despine(ax = ax[i_sess])
    ax[0].set_ylabel("Counts")
    ax[0].set_xlabel("Cluster")
    fig.tight_layout()

    return {plot_name: fig}

defaults = dict()