import numpy as np
import networkx as nx
from graphdisambiguation import create_UMAP_graph_and_embedding, GraphDisambiguation
import joblib
from collections import defaultdict
from tqdm import tqdm
from helpers import save_for_drawing, render_images
import os
import matplotlib.pyplot as plt
from matplotlib import cm

"""
This script generates all separate components used to construct Fig 6 in the paper.
"""

dataset = "mnist_testset_pca_200_norm" #code on how the data was generated can be found in mnist_pca_data.py
X = np.load(f"data/{dataset}/X.npy")
y = np.load(f"data/{dataset}/y.npy")
n_neighb = 15
epsilon = 0.7
JL = 2.0
CC = 2.0
distance_metric = 'euclidean'
weight_ratio = 0.05

UMAP_csr_original, UMAP_embedding_original = create_UMAP_graph_and_embedding(X, n_neighbors=n_neighb, metric=distance_metric)
UMAP_graph_original = nx.from_scipy_sparse_array(UMAP_csr_original)
gd = GraphDisambiguation(X,
                         y,
                         UMAP_graph_original,
                         UMAP_csr_original,
                         epsilon,
                         weight_ratio,
                         jl = JL,
                         cc = CC)
gd.fit()
graph_dict, X_extended_dict, y_extended_dict, split_dict, coverage_dict = gd.get_disambiguation_results()

new_embeddings_per_r = defaultdict(list)
for r in range(1, max(graph_dict.keys())+1):
    new_embedding = gd.create_disambiguated_UMAP_embedding(UMAP_embedding_original, distance_metric, r)
    new_embeddings_per_r[r] = new_embedding

save_for_drawing(UMAP_embedding_original,
                y,
                new_embeddings_per_r[2],
                y_extended_dict[2],
                split_dict[2],
                pred_labels = None

)
render_images('fig_6')

#Get the nearest neighbor(s)
nodes_to_process = [4126]
images = np.load(f"data/{dataset}/images.npy")

SAVE_DIR = "fig_6"
os.makedirs(SAVE_DIR, exist_ok=True)

results = {}
pca = joblib.load(f"data/{dataset}/pca.pkl")

for i in nodes_to_process:
    x_i_orig = images[i]
    x_i_pca  = pca.inverse_transform(X[i]).reshape(28, 28)
    label_i  = y[i]

    instance_dir = os.path.join(SAVE_DIR, f"i_{i}")
    os.makedirs(instance_dir, exist_ok=True)

    plt.imsave(os.path.join(instance_dir, f"i_{i}_original.png"),
               x_i_orig, cmap="gray")
    plt.imsave(os.path.join(instance_dir, f"i_{i}_pca.png"),
               x_i_pca, cmap="gray")

    results[i] = {
        "label": label_i,
        "original": x_i_orig,
        "pca": x_i_pca,
        "components": {}
    }

    ego = nx.ego_graph(gd.sparsified_graph, i, radius=1)
    G_removed = ego.copy()
    G_removed.remove_node(i)

    components = list(nx.connected_components(G_removed))

    for comp_idx, comp_nodes in enumerate(components):

        comp_nodes = list(comp_nodes)
        if len(comp_nodes) == 0:
            continue

        results[i]["components"][comp_idx] = {
            "nodes": {},
            "size": len(comp_nodes)
        }

        for node in comp_nodes:
            x_n_orig = images[node]
            x_n_pca  = pca.inverse_transform(X[node]).reshape(28, 28)

            dist = float(np.linalg.norm(x_i_pca.flatten() - x_n_pca.flatten()))

            # difference map
            diff_map = (x_i_orig - x_n_orig) * (x_i_pca - x_n_pca)
            diff_norm = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-12)
            viridis_rgb = cm.get_cmap("viridis")(diff_norm)[..., :3]

            if node == 8396 or node == 2246:
                base = f"i_{i}_component_{comp_idx}_node_{node}"

                plt.imsave(os.path.join(instance_dir, f"{base}_original.png"),
                        x_n_orig, cmap="gray")

                plt.imsave(os.path.join(instance_dir, f"{base}_pca.png"),
                        x_n_pca, cmap="gray")

                plt.imsave(os.path.join(instance_dir, f"{base}_viridis.png"),
                        viridis_rgb)

            results[i]["components"][comp_idx]["nodes"][node] = {
                "distance": dist
            }
    print(results[i]['components'])