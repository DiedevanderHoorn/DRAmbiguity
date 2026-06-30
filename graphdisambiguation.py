import umap 
import numpy as np
from sklearn.utils import check_random_state
import networkx as nx
from collections import defaultdict
import networkx as nx
import numpy as np
from collections import defaultdict
import igraph as ig
import subprocess
import uuid
import scipy.sparse as sp
import numpy as np
import os
import itertools

# Helper functions ##########################################################################################
def sparsify_with_julia(A_csr, epsilon, jlfac, matrixconcconst, julia_script="run_sparsify.jl"):
    A = A_csr

    in_file  = f"_tmp_in_{uuid.uuid4().hex}.npz"
    out_file = f"_tmp_out_{uuid.uuid4().hex}.npz"

    # Save CSC bundle
    save_csc_npz(in_file, A)

    # Run Julia
    subprocess.run(["julia", julia_script, in_file, out_file, str(epsilon), str(matrixconcconst), str(jlfac)], check=True)

    # Load CSC result
    As = load_csc_npz(out_file)

    os.remove(in_file)
    os.remove(out_file)
    return As

def save_csc_npz(path, A_csr):
    A_csc = A_csr.tocsc().astype(np.float64, copy=False)

    np.savez(
        path,
        data=A_csc.data,           # float64
        indices=A_csc.indices,     # row indices (0-based)
        indptr=A_csc.indptr,       # col pointers (0-based)
        shape=A_csc.shape          # (m, n)
    )


def load_csc_npz(path):
    D = np.load(path, allow_pickle=False)
    return sp.csc_matrix((D["data"], D["indices"], D["indptr"]), shape=tuple(D["shape"]))

def create_UMAP_graph_and_embedding(X, n_neighbors, metric): 
    """
    Construct UMAP embedding and return its graph.
    """

    mapper = umap.UMAP(n_neighbors = n_neighbors,
                    random_state = 42,
                    metric = metric,
                    spread=1.0, min_dist=0.1)
    mapper.fit(X)
    csr_X = mapper.graph_
    embedding = mapper.embedding_
    return csr_X, embedding

class GraphDisambiguation:
    def __init__(self, X, y, graph, csr, epsilon = 0.3, weight_ratio = 0.0, sparsified_graph=None, jl=4.0, cc=4.0):
        self.X_original = X.copy()
        self.y_original = y.copy()
        self.original_graph = graph.copy()
        self.csr = csr
        self.sparsified_graph = sparsified_graph
        
        self.seed = 42
        self.jl = jl
        self.cc = cc

        # Results
        self.X_per_radius = {}
        self.y_per_radius = {}
        self.graph_per_radius = {}
        self.splits_per_radius = {}
        self.avg_coverage_per_r = None
        
        self.epsilon = epsilon
        self.weight_ratio = weight_ratio
        self._alg_executed = False
        self._LAP_exist = True
        self._components_to_split_to_per_node_per_r = dict(dict())
        self._LAP_dict = None

    def fit(self):
        """
        Executes full disambiguation pipeline.
        Performs sparsification if needed, then identifies LAPs (ambiguous instances),
        computes relevant splits, and builds disambiguated graphs.
        """
        if self.sparsified_graph is None:
            self._jacc_scores = self.sparsify_graph()
            LAP_dict = self._find_r_neighborhood_articulation_points()
        if self.sparsified_graph is not None:
            LAP_dict = self._find_r_neighborhood_articulation_points()

        self._LAP_dict = LAP_dict
        
        if len(self._LAP_dict.get(1, [])) < len(self._LAP_dict.get(2, [])) or len(self._LAP_dict.get(1, [])) < len(self._LAP_dict.get(3, [])):
             raise ValueError(f"Problem in LAP identification")
        
        #Coverage statistics
        avg_coverage_per_r = {}
        for r, nodes_dict in LAP_dict.items():
            coverages = [info['coverage'] for info in nodes_dict.values()]
            avg_coverage_per_r[r] = sum(coverages) / len(coverages) if coverages else 0.0
        self.avg_coverage_per_r = avg_coverage_per_r

        #Splitting logic
        for r in LAP_dict.keys():
            components_to_split_to_per_node = self._determine_relevant_splits_at_r(r, LAP_dict)
            self._components_to_split_to_per_node_per_r[r] = components_to_split_to_per_node

            # Create split graphs
            self._duplicate_nodes_with_networkX(components_to_split_to_per_node, LAP_dict, r)
        
        self._alg_executed = True

    def get_disambiguation_results(self, radius=None):
        """
        Retrieve disambiguated graph, X, y, splits and coverage.
        """
       
        if not self._LAP_exist:
            raise ValueError(f"Under this parameterization there are no LAPs found. Try decreasing the neighborhood size or increasing the filtering threshold.")
        if not self._alg_executed:
            raise ValueError(f"Results not computed. Call fit() first.")
        if radius:
            if radius not in self.graph_per_radius.keys():
                raise ValueError(f"Radius {radius} does not have any results. Available radii: {self.graph_per_radius.keys()}.")
            return (
                self.graph_per_radius[radius], 
                self.X_per_radius[radius], 
                self.y_per_radius[radius], 
                self.splits_per_radius[radius],
                self.avg_coverage_per_r[radius]
            )
        else:
            return (
                self.graph_per_radius, 
                self.X_per_radius, 
                self.y_per_radius, 
                self.splits_per_radius,
                self.avg_coverage_per_r
            )
        

    def _find_r_neighborhood_articulation_points(self):
        """Compute local articulation point structure for all nodes over r."""
        G_ig = ig.Graph.from_networkx(self.sparsified_graph)
        G_ig.vs["name"] = list(self.sparsified_graph.nodes()) #to ensure no indexing issues

        n_nodes = G_ig.vcount()
        LAP_info_per_radius = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "direct_neighbors": set(),
                    "components_at_r": [],
                    "weight_per_component": [],
                    "coverage": None
                }
            )
        )

        for v in range(n_nodes):
            # Original ID of v
            v_id = G_ig.vs[v]["name"]

            # Full connected component of v
            comp_nodes = set(G_ig.neighborhood(v, order=n_nodes))
            comp_size = len(comp_nodes)

            for r in range(1, comp_size + 1):
                # Ego neighborhood including center
                ego_nodes = set(G_ig.neighborhood(v, order=r))
                direct_neighbors = set(G_ig.neighbors(v))

                # Induce subgraph
                ego_subgraph = G_ig.subgraph(list(ego_nodes))

                v_sub_idx = ego_subgraph.vs.find(name=v_id).index
                ego_subgraph.delete_vertices(v_sub_idx)

                if ego_subgraph.vcount() == 0:
                    break  # no nodes left

                # Compute connected components
                membership = ego_subgraph.connected_components().membership
                subgraph_nodes = [v["name"] for v in ego_subgraph.vs]  # original IDs
                comp_map = defaultdict(set)
                for idx, comp_id in enumerate(membership):
                    comp_map[comp_id].add(subgraph_nodes[idx])
                components = list(comp_map.values())

                # Neighbors per component to later know how to split
                neighbors_per_component = [
                    comp & direct_neighbors
                    for comp in components
                    if comp & direct_neighbors
                ]

                is_articulation = len(neighbors_per_component) > 1
                
                # Stop if point is not articulation; points cannot later become articulation points
                if not is_articulation:
                    break

                # Compute weights per component, can be used when determining splits
                weights_per_component = [
                    [
                        G_ig.es[G_ig.get_eid(v, n)]["weight"] if G_ig.are_adjacent(v, n) else 1
                        for n in (comp & direct_neighbors)
                    ]
                    for comp in components
                    if comp & direct_neighbors
                ]

                LAP_info_per_radius[r][v_id]['weight_per_component'] = weights_per_component
                LAP_info_per_radius[r][v_id]['coverage'] = len(ego_nodes) / n_nodes
                LAP_info_per_radius[r][v_id]['direct_neighbors'] = direct_neighbors
                LAP_info_per_radius[r][v_id]['components_at_r'] = neighbors_per_component

                # Stop if ego covers full component
                if ego_nodes == comp_nodes:
                    break

        return dict(LAP_info_per_radius)

    def _determine_relevant_splits_at_r(self, radius, LAP_dict):
        """
        Takes the LAP dict and determines the splits based on the splitting rules as detailed in the paper.
        Also ensures that there are no issues with indexing for the splits.
        """
        articulation_points = list(LAP_dict[radius].keys())
        ap_set = set(articulation_points)

        components_to_split_to_per_node = dict()

        for node, info in LAP_dict[radius].items():
            components_to_keep = []
            node = int(node)

            neighbors_per_component = info['components_at_r']
            weights_per_component = info['weight_per_component']

            component_sums = [sum(wlist) for wlist in weights_per_component]

            for i, comp in enumerate(neighbors_per_component):
                non_lap_nodes = {n for n in comp if n not in ap_set}
                if not non_lap_nodes:  # skip if all nodes were LAPs
                    continue
                
                non_lap_indices = [i for i, n in enumerate(comp) if n not in ap_set]
                filtered_weights = np.sum([weights_per_component[i][j] for j in non_lap_indices])

                if len(non_lap_nodes) == 1:
                   continue  
                
                if filtered_weights / max(component_sums) < self.weight_ratio:
                    continue
                
                components_to_keep.append(non_lap_nodes)

            if len(components_to_keep) > 1:
                components_to_split_to_per_node[node] = components_to_keep
        
        return components_to_split_to_per_node
    

    def _replace_with_duplicates(self, graph_dict):
        """
        Adjust neighbor references after duplicating nodes.
        """
        group_of = {}
        for base, inner_dict in graph_dict.items():
            for dup in inner_dict:
                group_of[dup] = base

        for base, inner_dict in graph_dict.items():
            for split_id, sets_list in inner_dict.items():
                for s in sets_list:
                    updated = set()
                    for n in s:
                        # If neighbor belongs to a split group, pick matching duplicate
                        if n in group_of:
                            neighbor_base = group_of[n]
                            neighbor_group = graph_dict[neighbor_base]

                            # If this is a duplicate, connect to neighbor's duplicate
                            # Otherwise, connect to neighbor's base
                            if split_id != base:
                                # pick the first duplicate (other than the base) if exists
                                possible = [k for k in neighbor_group if k != neighbor_base]
                                target = possible[0] if possible else n
                            else:
                                target = neighbor_base
                            updated.add(target)
                        else:
                            updated.add(n)

                    # Update the set in place
                    s.clear()
                    s.update(updated)

        return graph_dict

    def _duplicate_nodes_with_networkX(self, components_to_split_to_per_node, LAP_dict, radius):
        """
        Actual graph disambiguation step, results in a graph with splits, as well as X and y that have been updated to reflect the graph. 
        Note that X should not be used as an input for DR, we use sparsified graph.
        """
        graph = self.sparsified_graph.copy()
        articulation_points = list(LAP_dict[radius].keys())
       
        graph.remove_edges_from([(u, v) for u, v in itertools.combinations(articulation_points, 2) if graph.has_edge(u, v)])
        split_mapping_to_component = defaultdict(lambda: defaultdict(list))
        new_id_to_old = dict()
        number_of_splits = defaultdict(int)
        new_rows_X = []
        new_rows_y = []
        X_extended_r = self.X_original
        y_extended_r = self.y_original

        #Determine all splits and their new indices
        current_id = graph.number_of_nodes() 
        for node, components in components_to_split_to_per_node.items():
            for i, c in enumerate(components):
                filtered_comp = {v for v in c if v not in components_to_split_to_per_node}
                new_node_id = node if i == 0 else current_id

                split_mapping_to_component[node][new_node_id].append(filtered_comp)
                new_id_to_old[new_node_id] = node
                number_of_splits[node] += 1
                
                if i > 0:
                    new_rows_X.append(self.X_original[node])
                    new_rows_y.append(self.y_original[node])
                    current_id += 1

        if new_rows_X:
            X_extended_r = np.vstack([X_extended_r, *new_rows_X])
        if new_rows_y:
            y_extended_r = np.append(y_extended_r, new_rows_y)

        split_mapping_to_component_updated = self._replace_with_duplicates(split_mapping_to_component)   

        #Execute splitting
        for node, d in split_mapping_to_component_updated.items():
            for new_node, neighbors_dict in d.items():
                # neighbors_dict[0] is a set, convert to list for consistent iteration
                neighbor_group = list(neighbors_dict[0])

                # Map neighbors to old indices
                group_indices = [new_id_to_old[n] if n in new_id_to_old else n for n in neighbor_group]

                old_idx = new_id_to_old[new_node]
                edges_to_remove = [(old_idx, n_old) if graph.has_edge(old_idx, n_old) else (n_old, old_idx) for n_old in group_indices]
                graph.remove_edges_from(edges_to_remove)
                for n, n_old in zip(neighbor_group, group_indices):
                    weight = self.get_edge_weight(self.original_graph, old_idx, n_old)
                    graph.add_edge(new_node, n, weight=weight)

        disambiguated_graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=0)
        
        self.X_per_radius[radius] = X_extended_r
        self.y_per_radius[radius] = y_extended_r
        self.graph_per_radius[radius] = disambiguated_graph
        self.splits_per_radius[radius] = new_id_to_old

    def get_edge_weight(self, g, a, b):
        return g[a][b]['weight'] if g.has_edge(a, b) else 0
   
    def sparsify_graph(self):
        #convert adjacency to julia sparse
        As = sparsify_with_julia(self.csr, epsilon=self.epsilon, matrixconcconst=self.cc, jlfac=self.jl)
        self.sparsified_graph = nx.from_scipy_sparse_array(As)
        print(f"Sparsification removed {self.original_graph.number_of_edges() - self.sparsified_graph.number_of_edges()} edges, before: {self.original_graph.number_of_edges()}, now: {self.sparsified_graph.number_of_edges()}")

    def create_disambiguated_UMAP_embedding(self, original_embedding, metric, radius=None):
        """ Construction of the UMAP embedding based on the positioning of a previous embedding and the
        disambiguated graph.
        
        """
        if not self._alg_executed:
            raise ValueError("Call fit() before computing embeddings.")

        embedding_dict = {}
        for r in self.X_per_radius:
            X_extended = self.X_per_radius[r]
            if len(X_extended) == len(self.X_original):
                embedding_dict[r] = original_embedding
            graph_with_duplicates = self.graph_per_radius[r]
            components_to_split_to_per_node = self._components_to_split_to_per_node_per_r[r]

            #determine positions for duplicated nodes
            new_positions = original_embedding.copy()
            new_rows = []

            for node, components in components_to_split_to_per_node.items():
                for i, component in enumerate(components):
                    embeddings = [original_embedding[v] for v in component if v not in components_to_split_to_per_node]
                    avg_pos = np.mean(embeddings, axis = 0)
                    if i == 0:
                        new_positions[node] = avg_pos
                    else:
                        new_rows.append(avg_pos)
            if new_rows:
                new_positions = np.vstack([new_positions, *new_rows])
                
            #create embedding
            a, b = umap.umap_.find_ab_params(spread=1.0, min_dist=0.1)
            csr_duplicated_graph = nx.to_scipy_sparse_array(graph_with_duplicates, nodelist=sorted(graph_with_duplicates.nodes()), weight='weight', format='csr')
            max_weight = csr_duplicated_graph.data.max()

            # Avoid division by zero
            if max_weight > 0:
                csr_duplicated_graph.data /= max_weight

            X_embedded_new, _ = umap.umap_.simplicial_set_embedding(
                data = X_extended,
                graph = csr_duplicated_graph,
                n_components = 2,
                initial_alpha = 1.0,
                a = a,
                b = b,
                gamma = 1.0,
                negative_sample_rate = 5,
                n_epochs = 200,
                init=new_positions,
                random_state=check_random_state(42),
                metric=metric,
                metric_kwds={},
                densmap=False,
                densmap_kwds={},
                output_dens=False,
                euclidean_output=True,
                parallel=False,
                verbose=False
            )

            embedding_dict[r] = X_embedded_new
        
        if radius:
            if radius not in embedding_dict.keys():
                raise ValueError(f"No results for radius {radius}.")
            res = embedding_dict[radius]
        else:
            res = embedding_dict

        return res
           
