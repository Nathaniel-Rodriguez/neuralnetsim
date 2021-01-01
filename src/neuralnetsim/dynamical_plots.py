__all__ = ["plot_power_law_distributions"]


import neuralnetsim
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib import cm
import powerlaw
from pathlib import Path
from typing import List


def plot_power_law_distributions(
        distributions: List[np.ndarray],
        distribution_labels: List[float],
        xlabel: str = "",
        ylabel: str = "",
        save_dir: Path = None,
        prefix: str = "",
        pdf: bool = True
):
    colors = seaborn.color_palette("flare", len(distributions))
    fig, ax1 = plt.subplots()
    for i in range(len(distributions)):
        x, y = neuralnetsim.eccdf(distributions[i])
        ax1.plot(x, y, color=colors[i], lw=2,
                 label=str(round(distribution_labels[i], 2)))
    left, bottom, width, height = [0.1, 0.1, 0.4, 0.4]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(distribution_labels,
             [powerlaw.Fit(dist).truncated_power_law.parameter1
              for dist in distributions])
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend()
    if pdf:
        plt.savefig(save_dir.joinpath(
            prefix + "_powerlaw.pdf"))
    else:
        plt.savefig(save_dir.joinpath(
            prefix + "_powerlaw.png"), dpi=300)
    plt.close()
    plt.clf()


def plot_bridge_results(sim_df: pd.DataFrame,
                        original_graph: nx.DiGraph,
                        window_size: int,
                        save_dir: Path,
                        prefix: str = ""):
    seaborn.lineplot(data=sim_df, x=r'$\mu$', y='flow', hue='grid_par')
    plt.axhline(0.0, c='black', ls='--')
    plt.savefig("{0}_flow_w{1}.png".format(prefix, str(window_size)), dpi=300)
    plt.close()
    plt.clf()
    seaborn.lineplot(data=sim_df, x=r'$\mu$', y='activity', hue='grid_par')
    plt.axhline(0.0, c='black', ls='--')
    plt.savefig("{0}_act_w{1}.png".format(prefix, str(window_size)), dpi=300)
    plt.close()
    plt.clf()

    # color example
    # colors = cm.plasma(np.linspace(0, 1, len(bfs)))
    # for i, bf in enumerate(bfs):
    #     x, y = neuralnetsim.eccdf(bf)
    #     plt.plot(x, y, c=colors[i], alpha=0.2)
