__all__ = ['plot_slice']


import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from statsmodels.distributions.empirical_distribution import ECDF


def plot_slice(graph: nx.DiGraph,
               color_key: str = "level2",
               save_dir: Path = Path.cwd(),
               prefix: str = ""):
    nx.draw_networkx_nodes(
        graph,
        nx.get_node_attributes(graph, "pos"),
        list(nx.get_node_attributes(graph, color_key).values()),
        cmap=cm.Set1
    )
    weights = np.array(list(nx.get_edge_attributes(graph, "weight").values()))
    np.log(weights, out=weights)
    edges = nx.draw_networkx_edges(
        graph,
        nx.get_node_attributes(graph, "pos"),
        arrowstyle="->",
        arrowsize=10,
        edge_color=weights,
        edge_vmax=np.amax(weights),
        edge_vmin=np.amin(weights),
        edge_cmap=cm.Greys,
        width=2,
    )
    ecdf = ECDF(weights)
    edge_alphas = ecdf(weights)
    for i in range(len(edge_alphas)):
        edges[i].set_alpha(edge_alphas[i])

    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Greys)
    pc.set_array(weights)
    plt.colorbar(pc)

    ax = plt.gca()
    ax.set_axis_off()
    plt.savefig(save_dir.joinpath(prefix + "_slice.pdf"))
    plt.close()
    plt.clf()
