import numpy as np
import networkx as nx
from graphdisambiguation import GraphDisambiguation, create_UMAP_graph_and_embedding
from helpers import save_for_drawing, render_images
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def show_neighbors(query_id, neighbor_ids, images, ids):
    qidx = id_to_index[query_id]
    neigh_indices = [id_to_index[n] for n in neighbor_ids]

    plt.figure(figsize=(18, 4))
    plt.suptitle(f"Query ID {query_id} and 5 nearest neighbors", fontsize=20)

    plt.subplot(1, 7, 1)
    plt.imshow(np.transpose(images[id_dict[qidx]], (1, 2, 0)))
    plt.title(f"Query\n{query_id}")
    plt.axis("off")

    for j, nidx in enumerate(neigh_indices):
        if nidx in id_dict.keys():
            nidx = id_dict[nidx]
        plt.subplot(1, 7, j + 2)
        plt.imshow(np.transpose(images[nidx], (1, 2, 0)))
        plt.title(f"NN{j+1}\nID {ids[nidx]}")
        plt.axis("off")

    plt.savefig(f"fig_1/{query_id}_neighbors")

dataset = "svhn_5000_latent" #svhn_model_data.py contains the code on how to produce this data from the raw SVHN data
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
from collections import defaultdict
from tqdm import tqdm
new_embeddings_per_r = defaultdict(list)

for r in tqdm(range(1, max(graph_dict.keys())+1)):
    new_embedding = gd.create_disambiguated_UMAP_embedding(UMAP_embedding_original, distance_metric, r)
    new_embeddings_per_r[r] = new_embedding

save_for_drawing(UMAP_embedding_original,
                y,
                new_embeddings_per_r[2],
                y_extended_dict[2],
                split_dict[2],
                pred_labels = np.load("data/svhn_5000_latent/preds.npy")

)
render_images('fig_1')

import json
with open('data/data.json', "r") as f:
    data = json.load(f)
dup_data = data["disambiguated_embedding"]

ids = np.array([d["id"] for d in dup_data])
coords = np.array([[d["x"], d["y"]] for d in dup_data])

id_to_index = {id_val: i for i, id_val in enumerate(ids)}

nbrs = NearestNeighbors(n_neighbors = 7)  
# 6 because first neighbor is the point itself
nbrs.fit(coords)

query_ids = [1877, 5003]
neighbors_dict = {}

for qid in query_ids:
    idx = id_to_index[qid]
    distances, indices = nbrs.kneighbors([coords[idx]])
    neighbor_indices = indices[0][1:]  # skip the point itself
    neighbor_ids = ids[neighbor_indices]
    neighbors_dict[qid] = neighbor_ids.tolist()

print("Nearest neighbors:")
for q, neigh in neighbors_dict.items():
    print(q, ":", neigh)

image_path = "data/svhn_5000_latent/images.npy"
images = np.load(image_path)

id_dict = {5003 : 1877, 
        1877 : 1877}

for qid in query_ids:
    show_neighbors(qid, neighbors_dict[qid], images, ids)
