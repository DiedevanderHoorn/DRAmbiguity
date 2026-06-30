# # Reproduce Figure 2D: Han UMAP colored by hematopoietic lineages
import os
import numpy as np
import pyreadr
import matplotlib.pyplot as plt
import networkx as nx
from graphdisambiguation_spectral_spars import GraphDisambiguation, create_UMAP_graph_and_embedding
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

" Code for how to get the data (commented out) + code to generate fig 8"
# # If on Windows, make R visible for rpy2 (AUCell loading only)
# os.environ.setdefault("R_HOME", r"C:\Program Files\R\R-4.5.2")
# os.environ["PATH"] += r";C:\Program Files\R\R-4.5.2\bin"

# #data can be found here: https://figshare.com/s/9c3a0136f12b97f1dadd?file=12389231

# # --- Load PCA (input to UMAP) ---
# pca = pyreadr.read_r("BM_BMcKit_PB_RData/pca_g2.RData")["pca"]
# pca = np.array(pca)

# # --- Load subset mask ---
# g2 = pyreadr.read_r("BM_BMcKit_PB_RData/g2.RData")["g2"].to_numpy().ravel()
# subset_idx = ~np.isnan(g2)

# # --- Load AUCell (requires rpy2 because it's an S4 object) ---
# import rpy2.robjects as ro
# from rpy2.robjects.packages import importr

# AUCell = importr("AUCell")
# ro.r('load("BM_BMcKit_PB_RData/cells_AUC.RData")')

# # Convert AUCell → AUC matrix
# auc_mat = ro.r("as.matrix(AUCell::getAUC(cells_AUC))")
# auc_mat = np.array(auc_mat)   # shape: (lineages × cells_full)

# # Select subset of cells
# auc_sub = auc_mat[:, subset_idx]

# # --- Determine cell lineage by argmax across gene sets ---
# lineage_idx = np.argmax(auc_sub, axis=0)

# np.save('data/rna_seq/X.npy', pca)
# np.save('data/rna_seq/y.npy', lineage_idx)

LF = 1.0
CC = 1.0
dataset = "rna_seq"
n_neighbors = 30
weight_ratio = 0.1
epsilon = 0.9

pca = np.load('data/rna_seq/X.npy')
lineage_idx = np.load('data/rna_seq/y.npy')

csr, embedding = create_UMAP_graph_and_embedding(pca, n_neighbors=n_neighbors, metric='correlation')

graph = nx.from_scipy_sparse_array(csr)
gd = GraphDisambiguation(pca, lineage_idx, graph, csr, epsilon=epsilon, weight_ratio=weight_ratio, cc=CC, jl=LF)
gd.sparsify_graph()
gd.fit()

graph_dict, X_extended_dict, y_extended_dict, split_dict, coverage_dict = gd.get_disambiguation_results()
new_embeddings_per_r = defaultdict(list)

for r in range(1, max(graph_dict.keys())+1):
    new_embedding = gd.create_disambiguated_UMAP_embedding(embedding, "correlation", r)
    new_embeddings_per_r[r] = new_embedding

r = 2

dup_edges = gd.splits_per_radius[r]
labels = gd.y_per_radius[r]
coverage = gd.avg_coverage_per_r

# now plot
xs = new_embeddings_per_r[r][:,0]
ys = new_embeddings_per_r[r][:,1]


unique_labels = sorted(set(gd.y_per_radius[2]))
symbols = [
    "cross" if n in gd.splits_per_radius[2] else "circle"
    for n in range(len(xs))
]

name_dict = {
    0 : "B cell" ,
    1 : 'Basophil', 
    2: 'Dendritic', 
    3: 'Eosinophil', 
    4: 'Erythrocyte', 
    5: 'Macrophage', 
    6: 'Mast cell', 
    7: 'Megakaryocyte', 
    8: 'MPP', 
    9: 'Neutrophil', 
    10: 'NK cell', 
    11: 'T cell'
}

#Colors to match original paper coloring
lineage_to_color = {
    "MPP":        "#08306B",   
    "Macrophage": "#FFFFB3",   
    "Neutrophil": "#BEBADA",   
    "Erythrocyte": "#B3DE69",  
    "B cell":     "#BC80BD",   
    "T cell":     "#FB8072",   
    "NK cell":    "#FDB462",   
}

DEFAULT_GRAY = "#D9D9D9"


point_colors = np.array([
    lineage_to_color.get(name_dict[l], DEFAULT_GRAY)
    for l in gd.y_per_radius[2]
])

symbol_map = {
    "circle": "o",
    "cross": "X"
}
markers = [symbol_map.get(sym, "o") for sym in symbols]


by_marker = defaultdict(list)
for i, m in enumerate(markers):
    by_marker[m].append(i)

all_markers = list(by_marker.keys())
non_cross = [m for m in all_markers if m != "X"]
plot_order = non_cross + (["X"] if "X" in all_markers else [])


plt.figure(figsize=(10, 7))

dup_edge_color = "#434343"    
dup_edge_alpha = 1.0

from collections import defaultdict

reverse = defaultdict(list)

for src, dst in dup_edges.items():
    reverse[dst].append(src)

paths = {}

#connect all duplicates
for val, sources in reverse.items():
    if len(sources) == 1:
        paths[val] = [sources[0], val]

    else:
        chain = [val]
        for s in sources:
            chain += [s, val]
        paths[val] = chain


for val, path in paths.items():
    xs_path = xs[path]
    ys_path = ys[path]

    plt.plot(
        xs_path, ys_path,
        color=dup_edge_color,
        alpha=dup_edge_alpha,
        linewidth=1.5,
        linestyle=(0, (4, 4)), 
        zorder=1
    )

for m in plot_order:
    idxs = by_marker[m]

    if m == "X":
        z = 6
        edgecolor = "black"
        linewidths = 1.0
        size = 120
    else:
        z = 2
        edgecolor = "none"
        linewidths = 0.0
        size = 14

    plt.scatter(
        xs[idxs],
        ys[idxs],
        c=point_colors[idxs],
        marker=m,
        s=size,
        edgecolor=edgecolor,
        linewidths=linewidths,
        alpha=0.9,
        zorder=z,
    )


from matplotlib.lines import Line2D

# Only display relevant legend items
labels_to_include =  {"MPP",
    "Macrophage",
    "Neutrophil",
    "Erythrocyte",
    "B cell",
    "T cell",
    "NK cell"}

legend_handles = []

for lab in unique_labels:
    lineage_name = name_dict[lab] 

    if lineage_name not in labels_to_include:
        continue  

    color = lineage_to_color.get(lineage_name, DEFAULT_GRAY)

    handle = Line2D(
        [0], [0],
        marker='o',
        color='none',
        markerfacecolor=color,
        markersize=10,
        label=lineage_name
    )
    legend_handles.append(handle)


n_cols = min(4, len(legend_handles)) 

fig = plt.gcf()

fig.legend(
    handles=legend_handles,
    loc="lower center",
    frameon=False,
    ncol=n_cols,
    title="Lineage",
    fontsize = 14,
    title_fontsize = 14
)

ax = plt.gca()
ax.set_axis_off()

plt.savefig("fig_8", dpi=300, bbox_inches="tight")

