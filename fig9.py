import numpy as np
from graphdisambiguation_spectral_spars import create_UMAP_graph_and_embedding, GraphDisambiguation
import pickle
import os
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

dataset = "svhn_5000_latent"
X = np.load(f"data/{dataset}/X.npy")
y = np.load(f"data/{dataset}/y.npy")
distance_metric = 'euclidean'

results = defaultdict(lambda: defaultdict(int))
for n_neighbors in tqdm([5, 10, 15, 30, 50]):
    UMAP_csr_original, UMAP_embedding_original = create_UMAP_graph_and_embedding(X, n_neighbors=n_neighbors, metric=distance_metric)

    weight_ratio = 0.05

    epsilon = 0.7
    LF = 2.0
    CC = 2.0

    filename = f"sparsified_graphs/{dataset}_UMAP_UMAP_{n_neighbors}_{weight_ratio}_{epsilon}_{LF}_{CC}_sparsified"
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            sparsified_graph = pickle.load(f)
    else: sparsified_graph = None

    UMAP_graph_original = nx.from_scipy_sparse_array(UMAP_csr_original)
    gd = GraphDisambiguation(X, y, UMAP_graph_original, UMAP_csr_original, epsilon=epsilon, weight_ratio=weight_ratio, sparsified_graph=sparsified_graph, cc=CC, jl=LF)
    gd.fit()

    with open(f"sparsified_graphs/{dataset}_UMAP_UMAP_{n_neighbors}_{weight_ratio}_{epsilon}_{LF}_{CC}_sparsified", "wb") as f:
         pickle.dump(gd.sparsified_graph, f)
         print("saved sparsified graph")

    graph_dict, X_extended_dict, y_extended_dict, split_dict, coverage_dict = gd.get_disambiguation_results()
    for r in tqdm(range(1, max(graph_dict.keys())+1)):
        results[r][n_neighbors] = len(set(split_dict[r].values()))

import matplotlib.pyplot as plt
import matplotlib.cm as cm

results = dict(results)

with open("results_neighb.pkl", "wb") as f:
    pickle.dump(results, f)


with open("results_neighb.pkl", "rb") as f:
    results = pickle.load(f)

import matplotlib as mpl

n_lines = 8
cmap = mpl.colormaps['plasma']
custom_colors = cmap(np.linspace(0, 0.9, n_lines))[::-1]

plt.figure(figsize=(10, 6))

all_xs = set()
for idx, (r, vals) in enumerate(results.items()):
    xs = np.array(list(vals.keys()))
    ys = np.array(list(vals.values()))

    all_xs.update(xs)
    color = custom_colors[idx]

    if r in [6,7, 8]:
        jitter = (idx - (len(results)-1)/2) * 0.15 
    else:
        jitter = 0

    plt.plot(
        xs + jitter, ys,
        marker='o',
        markersize=4,
        linewidth=2,
        color=color,
        label=f"{r}",
        alpha=0.85
    )

plt.xlabel("Number of neighbors", fontsize=20)
plt.ylabel("Number of split ambiguous instances", fontsize=20)

all_xs = sorted(all_xs)
plt.xticks(all_xs, fontsize=18)
plt.yticks(fontsize = 18)

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.legend(title="Radius r",
           frameon=False,
           fontsize=18,
           title_fontsize=20)

plt.tight_layout()
plt.savefig('fig_9.png', dpi=300)
