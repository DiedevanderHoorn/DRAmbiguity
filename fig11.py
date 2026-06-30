import numpy as np
import matplotlib.pyplot as plt
from tensorflow_projection_qm.metrics import trustworthiness, continuity
import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
import os
from graphdisambiguation import create_UMAP_graph_and_embedding, GraphDisambiguation

def get_local_trustworthiness(X, embedding, n_neighbors):
    return trustworthiness.trustworthiness_with_local(X, embedding, k=n_neighbors)[1].numpy()

def get_local_continuity(X, embedding, n_neighbors):
    return continuity.continuity_with_local(X, embedding, k=n_neighbors)[1].numpy()

def draw_highlighted_curve(full_array, split_dict_at_r, LAP_dict_at_r, yaxis, filename=None, ax=None, highlight_color='#FDB462', ymin=None, ymax=None):
    sorted_indices = np.argsort(full_array)
    sorted_Ti = full_array[sorted_indices]
    x_sorted = np.arange(len(full_array))

    highlight_x = [
        i for i, orig_idx in enumerate(sorted_indices)
        if orig_idx in split_dict_at_r.keys()
    ]
    highlight_y = sorted_Ti[highlight_x]

    highlight_x_2 = [
        i for i, orig_idx in enumerate(sorted_indices)
        if orig_idx in LAP_dict_at_r.keys()
    ]
    highlight_y_2 = sorted_Ti[highlight_x_2]

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(x_sorted, sorted_Ti, s=12, alpha=1.0, label='Unsplit point', color="#80B1D3")
    ax.scatter(
        highlight_x_2,
        highlight_y_2,
        s=12,
        color=highlight_color,
        label='Identified as LAP, not split'
    )
    ax.scatter(
        highlight_x,
        highlight_y,
        s=40,
        marker='^',
        color=highlight_color,
        label='Point with at least 1 split',
        edgecolor='black',
        linewidth=1.5
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks([])
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize = 14)

    ax.set_xlabel("Instances", fontsize = 16)
    ax.set_ylabel(yaxis, fontsize = 16)
    ax.legend(loc='lower right', frameon=False, fontsize = 14)

    if filename:
        ax.figure.savefig(filename, dpi=300, bbox_inches='tight')


dataset = "mnist_testset_pca_200_norm"
X = np.load(f"data/{dataset}/X.npy")
y = np.load(f"data/{dataset}/y.npy")
distance_metric = "euclidean"

n_neighbors = 15
weight_ratio = 0.05
epsilon = 0.7
LF = 2.0
CC = 2.0

UMAP_csr_original, UMAP_embedding_original = create_UMAP_graph_and_embedding(X, n_neighbors=n_neighbors, metric=distance_metric)
UMAP_graph_original = nx.from_scipy_sparse_array(UMAP_csr_original)
gd = GraphDisambiguation(X, y, UMAP_graph_original, UMAP_csr_original, epsilon=epsilon, weight_ratio=weight_ratio, cc=CC, jl=LF)
gd.fit()

tw = get_local_trustworthiness(X, UMAP_embedding_original, n_neighbors)
cont = get_local_continuity(X, UMAP_embedding_original, n_neighbors)

os.makedirs("fig_11", exist_ok=True)
draw_highlighted_curve(tw, gd.splits_per_radius[2], gd._LAP_dict[2], "Trustworthiness", "fig_11/trustworthiness")
draw_highlighted_curve(cont, gd.splits_per_radius[2], gd._LAP_dict[2], "Continuity", "fig_11/continuity")