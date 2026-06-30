import numpy as np
from graphdisambiguation import GraphDisambiguation, create_UMAP_graph_and_embedding
import networkx as nx
from helpers import render_images, save_for_drawing
import os

dataset = "cbr-ilp-ir-son"
X = np.load(f"data/{dataset}/X.npy")
y = np.load(f"data/{dataset}/y.npy")
n_neighb = 10
epsilon = 0.7
JL = 1.0
CC = 1.0
distance_metric = 'cosine'
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
render_images('fig_7')

text_folder = "data/cbr_data/" 

import json
with open('data.json', "r") as f:
    data = json.load(f)
dup_data = data["disambiguated_embedding"]

ids = np.array([d["id"] for d in dup_data])
coords = np.array([[d["x"], d["y"]] for d in dup_data])

id_to_index = {id_val: i for i, id_val in enumerate(ids)}
id_to_instance = {d["id"]: d["instance"] for d in dup_data}

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors = 7)  
# 6 because first neighbor is the point itself
nbrs.fit(coords)

query_ids = [378, 676]
neighbors_dict = {}

for qid in query_ids:
    idx = id_to_index[qid]
    distances, indices = nbrs.kneighbors([coords[idx]])
    neighbor_indices = indices[0][1:]  # skip the point itself
    neighbor_ids = ids[neighbor_indices]
    neighbors_dict[qid] = neighbor_ids.tolist()

print("Nearest neighbors:")
for q, neigh in neighbors_dict.items():
    print(q, "→", neigh)


output_file = "fig_7/nearest_neighbors.txt"

with open(output_file, "w", encoding="latin1") as out_f:
    for qid in query_ids:
        qinst = id_to_instance[qid]
        out_f.write(f"=== QUERY ID {qid}, Instance: {qinst} ===\n\n")

        qfile_path = os.path.join(text_folder, qinst)
        if os.path.exists(qfile_path):
            with open(qfile_path, "r", encoding="latin1", errors = "ignore") as f:
                out_f.write(f"--- QUERY FILE CONTENT ---\n")
                out_f.write(f.read() + "\n\n")
        else:
            out_f.write("Query file not found.\n\n")

        out_f.write("--- NEIGHBORS ---\n")
        for nn_id, nn_inst in neighbors_dict[qid]:
            nn_file_path = os.path.join(text_folder, nn_inst)
            out_f.write(f"Neighbor ID {nn_id}, Instance: {nn_inst}\n")
            if os.path.exists(nn_file_path):
                with open(nn_file_path, "r", encoding="latin1") as f:
                    out_f.write(f.read() + "\n\n")
            else:
                out_f.write("File not found.\n\n")
        out_f.write("="*80 + "\n\n")

print(f"All texts written to {output_file}")
