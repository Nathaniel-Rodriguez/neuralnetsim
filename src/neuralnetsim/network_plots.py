__all__ = ['plot_slice',
           "plot_ccdf_distributions",
           "plot_graph_strength_distributions"]


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn
from matplotlib import cm
from pathlib import Path
from statsmodels.distributions.empirical_distribution import ECDF
from typing import List
from neuralnetsim import calc_strength_distribution
from neuralnetsim import calc_nodal_strength_difference_distribution


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


def plot_ccdf_distributions(original_dist: np.ndarray,
                            comparison_dists: List[np.ndarray],
                            xlabel: str = "",
                            ylabel: str = "",
                            xscale: str = "linear",
                            yscale: str = "linear",
                            original_legend_label: str = "original",
                            comparison_legend_label: str = "comparison",
                            save_dir: Path = Path.cwd(),
                            prefix: str = ""):
    for dist in comparison_dists:
        ecdf = ECDF(dist)
        x = np.sort(dist)
        y = 1.0 - ecdf(x)
        plt.plot(x, y, c="grey", alpha=0.5, lw=0.5)

    ecdf = ECDF(original_dist)
    x = np.sort(original_dist)
    y = 1.0 - ecdf(x)
    plt.plot(x, y, c=seaborn.color_palette("tab10")[0], lw=2)
    alt_patch = mpatches.Patch(color='grey', label=comparison_legend_label)
    orig_patch = mpatches.Patch(color=seaborn.color_palette("tab10")[0],
                                label=original_legend_label)
    plt.legend(handles=[orig_patch, alt_patch])
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_dir.joinpath(
        prefix + "_ccdf.pdf"))


def plot_graph_strength_distributions(original_graph: nx.DiGraph,
                                      comparison_graphs: List[nx.DiGraph],
                                      direction="in",
                                      save_dir: Path = Path.cwd(),
                                      prefix: str = ""):
    original_dist = calc_strength_distribution(original_graph, direction)
    comparison_dists = [calc_strength_distribution(alt_graph, direction)
                        for alt_graph in comparison_graphs]
    plot_ccdf_distributions(original_dist,
                            comparison_dists,
                            "weight",
                            "CCDF",
                            "log",
                            "linear",
                            "Original",
                            "Generated",
                            save_dir,
                            prefix + "_" + direction + "_strength")


def plot_graph_nodal_strength_differences(original_graph: nx.DiGraph,
                                          comparison_graphs: List[nx.DiGraph],
                                          save_dir: Path = Path.cwd(),
                                          prefix: str = ""):
    original_dist = calc_nodal_strength_difference_distribution(original_graph)
    comparison_dists = [calc_nodal_strength_difference_distribution(alt_graph)
                        for alt_graph in comparison_graphs]
    plot_ccdf_distributions(original_dist,
                            comparison_dists,
                            "weight",
                            "CCDF",
                            "linear",
                            "linear",
                            "Original",
                            "Generated",
                            save_dir,
                            prefix + "_" + "_sdiff")
