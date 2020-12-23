__all__ = ["plot_power_law_distributions"]


import neuralnetsim
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib import cm
from pathlib import Path
from typing import List


def plot_power_law_distributions(
        distributions: np.ndarray,
        distribution_labels: List[str],
        xlabel: str = "",
        ylabel: str = "",
        save_dir: Path = None,
        prefix: str = "",
        pdf: bool = True
):
    colors = seaborn.color_palette("flare", len(distributions))
    for i in range(len(distributions)):
        ecdf = ECDF(distributions[i])
        x = np.sort(ecdf)
        y = 1.0 - ecdf(x)
        plt.plot(x, y, color=colors[i], lw=2, label=distribution_labels[i])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if pdf:
        plt.savefig(save_dir.joinpath(
            prefix + "_powerlaw.pdf"))
    else:
        plt.savefig(save_dir.joinpath(
            prefix + "_powerlaw.png"), dpi=300)
    plt.close()
    plt.clf()


def plot_simulation_results(pd):
    pass