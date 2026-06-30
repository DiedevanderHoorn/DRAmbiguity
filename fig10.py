import os
import numpy as np
import matplotlib.pyplot as plt
from graphdisambiguation import create_UMAP_graph_and_embedding, GraphDisambiguation
import os
import pickle
import umap
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from tqdm import tqdm
import networkx as nx

dataset_with_params =  {
     "cbr-ilp-ir-son" : {
            'n_neighb' : 10,
            'epsilon' : 0.7,
            'LF' : 1.0,
            'CC' : 1.0,
            'distance_metric' : 'cosine',
            'weight_ratio' : 0.05
        },

        "mnist_testset_pca_200_norm" : {
            'n_neighb' : 15,
            'epsilon' : 0.7,
            'LF' : 2.0,
            'CC' : 2.0,
            'distance_metric' : 'euclidean',
            'weight_ratio' : 0.05
        },

        "rna_seq" : {
            'n_neighb' : 30,
            'epsilon' : 0.9,
            'LF' : 1.0,
            'CC' : 1.0,
            'distance_metric' : 'correlation',
            'weight_ratio' : 0.1
        },

         "svhn_5000_latent" : {
            'n_neighb' : 15,
            'epsilon' : 0.7,
            'LF' : 2.0,
            'CC' : 2.0,
            'distance_metric' : 'euclidean',
            'weight_ratio' : 0.05
        }
       
    }

def compare_neighborhood_overlap_embeddings(A, B, n_neighbors):
    nn_A = NearestNeighbors(n_neighbors=n_neighbors).fit(A).kneighbors(return_distance=False)
    nn_B = NearestNeighbors(n_neighbors=n_neighbors).fit(B).kneighbors(return_distance=False)

    scores = []
    for i in range(len(A)):
        common = set(nn_A[i]) & set(nn_B[i])
        scores.append(len(common) / n_neighbors)

    return scores

def compare_neighborhood_overlap_high_low(high, low, distance_metric, n_neighbors):
    nn_high = NearestNeighbors(n_neighbors=n_neighbors, metric=distance_metric).fit(high).kneighbors(return_distance=False)
    nn_low = NearestNeighbors(n_neighbors=n_neighbors).fit(low).kneighbors(return_distance=False)

    scores = []
    for i in range(len(high)):
        common = set(nn_high[i]) & set(nn_low[i])
        scores.append(len(common) / n_neighbors)
    
    return scores

for dataset, param_dict in tqdm(dataset_with_params.items()):
    param_dict = dataset_with_params[dataset]
    n_neighbors = param_dict['n_neighb']
    epsilon = param_dict['epsilon']
    CC = param_dict['CC']
    LF = param_dict['LF']
    distance_metric = param_dict['distance_metric']
    weight_ratio = param_dict['weight_ratio']

    X = np.load(f"data/{dataset}/X.npy")
    y = np.load(f"data/{dataset}/y.npy")

    UMAP_csr_original, _ = create_UMAP_graph_and_embedding(X, n_neighbors=n_neighbors, metric=distance_metric)
    UMAP_graph_original = nx.from_scipy_sparse_array(UMAP_csr_original)
    gd = GraphDisambiguation(X, y,UMAP_graph_original, UMAP_csr_original, epsilon, weight_ratio)
    gd.sparsify_graph()
    sparsified_graph = nx.to_scipy_sparse_array(gd.sparsified_graph, dtype=float)
    max_weight = sparsified_graph.data.max()
    # Avoid division by zero
    if max_weight > 0:
        sparsified_graph.data /= max_weight

    master_rng = check_random_state(42)

    embeddings_full = []
    embeddings_sparse = []

    for i in tqdm(range(5)):
        sparse_graph_copy = sparsified_graph
        rng = check_random_state(master_rng.randint(0, 2**31 - 1))
        a, b = umap.umap_.find_ab_params(spread=1.0, min_dist=0.1)
        X_embedded_full, _ = umap.umap_.simplicial_set_embedding(
            data = X,
            graph = UMAP_csr_original,
            n_components = 2,
            initial_alpha = 1.0,
            a = a,
            b = b,
            gamma = 1.0,
            negative_sample_rate = 5,
            n_epochs = 200,
            init='spectral',
            random_state=rng,
            metric=distance_metric,
            metric_kwds={},
            densmap=False,
            densmap_kwds={},
            output_dens=False,
            euclidean_output=True,
            parallel=False,
            verbose=False
        )


        X_embedded_sparse, _ = umap.umap_.simplicial_set_embedding(
            data = X,
            graph = sparse_graph_copy,
            n_components = 2,
            initial_alpha = 1.0,
            a = a,
            b = b,
            gamma = 1.0,
            negative_sample_rate = 5,
            n_epochs = 200,
            init='spectral',
            random_state=rng,
            metric=distance_metric,
            metric_kwds={},
            densmap=False,
            densmap_kwds={},
            output_dens=False,
            euclidean_output=True,
            parallel=False,
            verbose=False
        )

        embeddings_full.append(X_embedded_full)
        embeddings_sparse.append(X_embedded_sparse)

    scores_mean_dict = dict()
    scores_dict = dict()

    #COMPARE 2 EMBEDDINGS FOR FULL GRAPH
    for (i,j) in combinations(range(len(embeddings_full)), 2):
        A = embeddings_full[i]
        B = embeddings_full[j]

        scores_full = compare_neighborhood_overlap_embeddings(A,B, n_neighbors)
        scores_dict.setdefault("full_full", []).append(scores_full)
        scores_mean_dict.setdefault("full_full", []).append(np.mean(scores_full))


    for i in range(len(embeddings_full)):
        for j in range(len(embeddings_sparse)):
            A = embeddings_full[i]
            B = embeddings_sparse[j]
            
            scores_sparse = compare_neighborhood_overlap_embeddings(A,B, n_neighbors)
            scores_dict.setdefault("full_sparse", []).append(scores_sparse)
            scores_mean_dict.setdefault("full_sparse", []).append(np.mean(scores_sparse))

    for i in range(len(embeddings_full)):
        scores_full_hd = compare_neighborhood_overlap_high_low(X, embeddings_full[i], distance_metric, n_neighbors)
        scores_dict.setdefault("full_high", []).append(scores_full_hd)
        scores_mean_dict.setdefault("full_high", []).append(np.mean(scores_full_hd))


    for i in range(len(embeddings_sparse)):
        scores_sparse_hd = compare_neighborhood_overlap_high_low(X, embeddings_sparse[i], distance_metric, n_neighbors)
        scores_dict.setdefault("sparse_high", []).append(scores_sparse_hd)
        scores_mean_dict.setdefault("sparse_high", []).append(np.mean(scores_sparse_hd))

    ratios = []
    for og_og in scores_mean_dict["full_full"]:
        for og_sparse in scores_mean_dict["full_sparse"]:
            ratios.append(min(1, og_sparse / og_og if og_og > 0 else 0))

    ratios_high = []
    for og_og in scores_mean_dict["full_high"]:
        for og_sparse in scores_mean_dict["sparse_high"]:
            ratios_high.append(min(1, og_sparse / og_og if og_og > 0 else 0))

    os.makedirs(f"neighborhood_overlap_results/{dataset}_{epsilon}", exist_ok=True)
    with open(f"neighborhood_overlap_results/{dataset}_{epsilon}/neighborhood_overlap_scores_p", "wb") as f:
        pickle.dump(scores_dict, f)

    with open(f"neighborhood_overlap_results/{dataset}_{epsilon}/neighborhood_overlap_means_p", "wb") as f:
        pickle.dump(scores_mean_dict, f)

    np.save(f"neighborhood_overlap_results/{dataset}_{epsilon}/ratios_spoverfull_p.npy", ratios)
    np.save(f"neighborhood_overlap_results/{dataset}_{epsilon}/ratios_high_spoverfull_p.npy", ratios_high)

BASE_DIR = "neighborhood_overlap_results"

datasets = sorted(os.listdir(BASE_DIR))
all_ratios = []
all_ratios_high = []
labels = []

# Load ratios for each dataset
for dataset in datasets:
    dataset_path = os.path.join(BASE_DIR, dataset)

    ratios_path = os.path.join(dataset_path, "ratios_spoverfull_p.npy")
    ratios_high_path = os.path.join(dataset_path, "ratios_high_spoverfull_p.npy")

    ratios = np.load(ratios_path, allow_pickle=True)
    ratios_high = np.load(ratios_high_path, allow_pickle=True)

    all_ratios.append(ratios)
    all_ratios_high.append(ratios_high)
    labels.append(dataset)


plt.rcParams.update({
    "font.size": 20,            
    "axes.titlesize": 20,      
    "axes.labelsize": 20,     
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 20,     
    "legend.title_fontsize": 18
})


labels = ["CBR", "MNIST", "RNA‑Seq", "SVHN" ]

pastel_set3 = [ "#80B1D3", "#FB8072"]
color1 = pastel_set3[0]
color2 = pastel_set3[1]

combined_data = []
for i in range(len(labels)):
    combined_data.append([all_ratios[i], all_ratios_high[i]])

fig, ax = plt.subplots(figsize=(8, 6))

# Create the positions for paired boxplots
positions = []
pos = 1
for _ in labels:
    positions.append((pos, pos + 0.4))
    pos += 1.5  

flat_positions = [p for pair in positions for p in pair]
flat_data = [d for pair in combined_data for d in pair]

colors = [color1, color2] * len(labels)

bp = ax.boxplot(
    flat_data,
    positions=flat_positions,
    widths=0.35,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=0.8)
    
)

for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_edgecolor("black")
plt.setp(bp["boxes"], linewidth=0.8)

#center ticks
tick_positions = [sum(p)/2 for p in positions]
ax.set_xticks(tick_positions)
ax.set_xticklabels(labels, rotation=0)

ax.set_ylabel("Ratio")

legend_labels = [
    r"$\rho_{2D}$",
    r"$\rho_{HD}$"
]
ax.legend(
    handles=[
        plt.Line2D([], [], color=color1, lw=10),
        plt.Line2D([], [], color=color2, lw=10),
    ],
    labels=legend_labels,
    frameon=False,
    title="Ratios",
    loc="lower right" )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("fig_10.png", dpi=200)

