__all__ = ['plot_slice']


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from statsmodels.distributions.empirical_distribution import ECDF


def plot_slice(graph: nx.DiGraph,
               color_key: str = "level2",
               save_dir: Path = Path.cwd(),
               prefix: str = ""):
    """
    Creates a pdf plot of the neural slice with nodes colored by community and
    positioned from data which requires the node "pos" attribute. Edges
    require the "weight" attribute.
    :param graph: A networkx graph with "pos" node attribute and edge "weight"
                  attributes.
    :param color_key: The community key for node coloring.
    :param save_dir: Directory to save the plot into.
    :param prefix: A prefix to prepend to the plot name.
    :return: None
    """
    nx.draw_networkx_nodes(
        graph,
        nx.get_node_attributes(graph, "pos"),
        node_color=list(nx.get_node_attributes(graph, color_key).values()),
        node_size=20,
        cmap=cm.Set1,
        label=None,
        linewidths=0.0
    )
    weights = np.array(list(nx.get_edge_attributes(graph, "weight").values()))
    np.log(weights, out=weights)
    ecdf = ECDF(weights)
    edge_alphas = ecdf(weights)
    for i, edge in enumerate(graph.edges):
        nx.draw_networkx_edges(
            graph,
            nx.get_node_attributes(graph, "pos"),
            edgelist=[edge],
            arrows=False,
            arrowstyle="->",
            arrowsize=1,
            edge_color=[weights[i]],
            edge_vmin=np.amin(weights),
            edge_vmax=np.amax(weights),
            edge_cmap=cm.Greys,
            width=0.5,
            alpha=edge_alphas[i]
        )

    ax = plt.gca()
    ax.set_axis_off()
    plt.savefig(save_dir.joinpath(prefix + "_slice.pdf"))
    plt.close()
    plt.clf()
