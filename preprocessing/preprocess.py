import numpy as np

import importlib
import networkx as nx
import torch
import torch.utils
import torch.utils.data


def _get_all_pairs_shortest_path(_G):
    """
    Pre-compute the all pair shortest paths of the assigned graph G
    """
    # Construct the all pair shortest path lookup
    if importlib.util.find_spec("networkit") is not None:
        import networkit as nk
        Gk = nk.nxadapter.nx2nk(_G)
        apsp = nk.distance.APSP(Gk).run().getDistances()
        lengths = {}
        for i, n1 in enumerate(_G.nodes()):
            lengths[n1] = {}
            for j, n2 in enumerate(_G.nodes()):
                if apsp[i][j] < 1e300:  # to drop unreachable node
                    lengths[n1][n2] = apsp[i][j]
        return lengths
    else:
        lengths = dict(nx.all_pairs_dijkstra_path_length(_G))
        return lengths


def compute_shortest_path(A_array, nx_g, graph_type='union_graph'):
    sub_graphs = []
    subgraph_nodes_list = []
    nx_g = nx_g.to_undirected()
    nx_g = nx_g.to_directed()
    for i in np.arange(len(A_array)):
        s_indexes = []
        for j in np.arange(len(A_array)):
            s_indexes.append(i)
            if (A_array[i][j] == 1):
                s_indexes.append(j)
        sub_graphs.append(nx_g.subgraph(s_indexes))

    for i in np.arange(len(sub_graphs)):
        subgraph_nodes_list.append(list(sub_graphs[i].nodes))

    weight = torch.zeros(nx_g.number_of_nodes(), nx_g.number_of_nodes())

    for u, v, e in nx_g.edges(data=True):
        source_nbr = np.nonzero(A_array[u])[0].tolist()
        target_nbr = np.nonzero(A_array[v])[0].tolist()
        source_nbr.append(u)
        target_nbr.append(v)

        union_g = nx_g.subgraph(source_nbr + target_nbr)
        _apsp = _get_all_pairs_shortest_path(union_g)

        # construct the cost dictionary from x to y
        if graph_type == 'union_graph':
            node_list = union_g.nodes()
            node_num = len(node_list)
            d = np.zeros((node_num, node_num))
            for k, src in enumerate(node_list):
                for j, dst in enumerate(node_list):
                    assert dst in _apsp[src], \
                        "Target node not in list, should not happened, pair (%d, %d)" % (src, dst)
                    d[k][j] = _apsp[src][dst]
        else:
            raise NotImplementedError

        _, s, _ = np.linalg.svd(d, full_matrices=True)
        sum_w = s.sum()

        weight[u][v] = sum_w
        weight[v][u] = sum_w

    return weight
