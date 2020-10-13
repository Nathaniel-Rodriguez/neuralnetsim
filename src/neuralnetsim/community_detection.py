__all__ = ["get_network", "add_communities", "add_positions",
           "build_graph_from_data"]


import infomap
import numpy as np
import networkx as nx
from pathlib import Path
from neuralnetsim.data_loader import load_as_matrix


def get_network(weight_matrix: np.ndarray,
                link_matrix: np.ndarray,
                **kwargs) -> nx.DiGraph:
    """
    Creates a networkx graph from loaded data.
    :param weight_matrix: A matrix of weights.
    :param link_matrix: A matrix of significant links.
    :return: A directed networkx graph.
    """
    adj_matrix = weight_matrix.copy()
    mask = np.array(link_matrix, dtype=np.bool)
    adj_matrix[np.where(~mask)] = 0
    return nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph)


def add_communities(graph: nx.DiGraph, seed=None,
                    infomap_commands=None, **kwargs) -> nx.DiGraph:
    """
    Uses Infomap to detect communities in a given graph and then assigns
    those communities as an attribute to the nodes of graph called "level1"
    for first depth level modules and "level2" for second depth level modules.
    :param graph: A weighted-directed networkx graph.
    :param seed: A seed for Infomap (default: None).
    :param infomap_commands: Optional command arguments for Infomap (default:
        ["-N 10", "--two-level", "--directed", "--silent"]).
    :return: Updates graph in-place with community assignments, returns the
             provided graph.
    """
    if infomap_commands is None:
        infomap_commands = ["-N 10", "--two-level", "--directed", "--silent"]
    if seed is not None:
        infomap_commands.append("--seed " + str(seed))
    imap = infomap.Infomap(" ".join(infomap_commands))
    imap.add_links(tuple((i, j, w) for i, j, w in graph.edges.data('weight')))
    imap.run()
    nx.set_node_attributes(graph, imap.get_modules(depth_level=1), "level1")
    nx.set_node_attributes(graph, imap.get_modules(depth_level=2), "level2")

    # give all community-less nodes a unique community
    unique_module_counter = max(
        max({m for i, m in graph.nodes.data("level1") if m is not None}),
        max({m for i, m in graph.nodes.data("level2") if m is not None})
    )
    lonely_nodes = {node: unique_module_counter + i
                    for i, node in enumerate({
                        n for n, m, in graph.nodes.data("level1") if m is None})}
    nx.set_node_attributes(graph, lonely_nodes, "level1")
    nx.set_node_attributes(graph, lonely_nodes, "level2")

    return graph


def add_positions(graph: nx.DiGraph, xpos: np.ndarray,
                  ypos: np.ndarray) -> nx.DiGraph:
    """
    Adds "pos" attributes to nodes from a position array.
    :param graph: A networkx graph to add position attributes too.
    :param xpos: A numpy array with x positions.
    :param ypos: A numpy array with y positions.
    :return: The provided graph, it is modified in-place.
    """
    # For some reason data is stored as a (1xN) array
    xpos = xpos.flatten()
    ypos = ypos.flatten()

    if len(xpos) != len(ypos):
        raise AttributeError("Unmatched x and y positions,"
                             " with {0} x positions and {1} y positions".format(
                                str(len(xpos)), str(len(ypos))))
    nx.set_node_attributes(graph, {i: (xpos[i], ypos[i])
                                   for i in range(len(xpos))}, "pos")
    return graph


def build_graph_from_data(data_dir: Path,
                          link_filename: str,
                          weight_filename: str,
                          pos_filename: str,
                          **kwargs) -> nx.DiGraph:
    """
    Creates a returns a weighted-directed networkx graph representing the loaded
    data. Includes node attributes of "level1" and "level2" for communities
    and "pos" for x,y coordinates. Includes edge attributes of "weight" for
    link weights.
    :param data_dir: Location of the data files.
    :param link_filename: Name of the file containing binary link matrix.
    :param weight_filename: Name of the file containing TE weights.
    :param pos_filename: Name of the file containing neuron x-y position.
    :param kwargs: Keyword arguments for get_network and add_communities.
    :return: A networkx DiGraph built from the data.
    """
    return add_positions(
        add_communities(
            get_network(
                load_as_matrix(data_dir.joinpath(weight_filename), "weights"),
                load_as_matrix(data_dir.joinpath(link_filename), "pdf"),
                **kwargs),
            **kwargs),
        load_as_matrix(data_dir.joinpath(pos_filename), "x"),
        load_as_matrix(data_dir.joinpath(pos_filename), "y")
    )
