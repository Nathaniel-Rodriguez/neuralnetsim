__all__ = ["plot_power_law_distributions",
           "plot_bridge_flow_contour",
           "plot_bridge_act_contour",
           "plot_bridge_slice",
           "plot_outflow",
           "plot_global_flow",
           "plot_com_flow",
           "plot_global_flow_contour"]


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


def plot_bridge_slice(
        sim_df: pd.DataFrame,
        original_graph: nx.DiGraph,
        window_size: int,
        save_dir: Path,
        prefix: str = ""):
    max_idx = sim_df['flow'].idxmax()
    max_par = sim_df['grid_par'][max_idx]
    max_df = sim_df[sim_df['grid_par'] == max_par]
    ax = seaborn.lineplot(data=max_df, x=r'$\mu$', y='flow', hue='grid_par', palette="tab10")
    ax.set_ylabel(r"$\Phi^{\mu}$")
    ax.get_legend().remove()
    plt.axhline(0.0, c='black', ls='--')
    plt.axvline(neuralnetsim.calc_mu(original_graph, "level1"), color="red")
    plt.tight_layout()
    plt.savefig(save_dir.joinpath("{0}_flow_cut.pdf".format(prefix)), dpi=300)
    plt.close()
    plt.clf()


def plot_outflow(
        outflow_df: pd.DataFrame,
        save_dir: Path,
        prefix: str = ""
):
    lg_coms = outflow_df[outflow_df['com'] < 6]  # largest 5 coms only
    ax = seaborn.lineplot(data=lg_coms, x='control_var', y='flow', hue='com', palette="tab10")
    ax.set_ylabel(r"$\Phi^{\mu}$")
    ax.set_xlabel(r"$\rho$")
    plt.tight_layout()
    plt.savefig(save_dir.joinpath("{0}_outflow.pdf".format(prefix)), dpi=300)
    plt.close()
    plt.clf()


def plot_global_flow(
        outflow_df: pd.DataFrame,
        save_dir: Path,
        prefix: str = ""
):
    seaborn.lineplot(data=outflow_df, x='control_var', y='flow')
    plt.savefig(save_dir.joinpath("{0}_globalflow.png".format(prefix)), dpi=300)
    plt.close()
    plt.clf()


def plot_com_flow(
        outflow_df: pd.DataFrame,
        save_dir: Path,
        prefix: str = ""
):
    seaborn.lineplot(data=outflow_df, x='control_var', y='flow', hue='com')
    plt.savefig(save_dir.joinpath("{0}_comflow.png".format(prefix)), dpi=300)
    plt.close()
    plt.clf()


def plot_bridge_flow_contour(
        sim_df: pd.DataFrame,
        original_graph: nx.DiGraph,
        save_dir: Path,
        prefix: str = "",
        n_trials: int = 30
):
    n_mus = sim_df[r'$\mu$'].nunique()
    n_par = sim_df["grid_par"].nunique()
    mus = sim_df[[r'$\mu$']].to_numpy()
    par = sim_df[["grid_par"]].to_numpy()
    flows = sim_df[["flow"]].to_numpy()
    mus = mus.reshape((n_par, n_mus, n_trials))
    pars = par.reshape((n_par, n_mus, n_trials))
    flows = flows.reshape((n_par, n_mus, n_trials))

    f, ax = plt.subplots(1, 1)
    cont = ax.contourf(mus[:, :, 0], pars[:, :, 0], np.mean(flows, axis=2))
    ax.axvline(neuralnetsim.calc_mu(original_graph, "level1"), color='red')
    ax.set_ylabel(r"$\rho$")
    ax.set_xlabel(r"$\mu$")
    f.colorbar(cont)
    plt.tight_layout()
    f.savefig(save_dir.joinpath(prefix + "_flow_contour.pdf"))
    plt.close()
    plt.clf()


def plot_global_flow_contour(
        sim_df: pd.DataFrame,
        original_graph: nx.DiGraph,
        save_dir: Path,
        prefix: str = "",
        n_trials: int = 30
):
    n_mus = sim_df[r'$\mu$'].nunique()
    n_par = sim_df["control_var"].nunique()
    mus = sim_df[[r'$\mu$']].to_numpy()
    par = sim_df[["control_var"]].to_numpy()
    flows = sim_df[["flow"]].to_numpy()
    mus = mus.reshape((n_par, n_mus, n_trials, -1))
    pars = par.reshape((n_par, n_mus, n_trials, -1))
    flows = flows.reshape((n_par, n_mus, n_trials, -1))
    flows = np.mean(flows, axis=3)

    f, ax = plt.subplots(1, 1)
    ax.contourf(mus[:, :, 0, 0], pars[:, :, 0, 0], np.mean(flows, axis=2))
    ax.axvline(neuralnetsim.calc_mu(original_graph, "level1"))
    f.savefig(save_dir.joinpath(prefix + "_global_contour.pdf"))
    plt.close()
    plt.clf()


def plot_bridge_act_contour(
        sim_df: pd.DataFrame,
        original_graph: nx.DiGraph,
        save_dir: Path,
        prefix: str = "",
        n_trials: int = 30
):
    n_mus = sim_df[r'$\mu$'].nunique()
    n_par = sim_df["grid_par"].nunique()
    mus = sim_df[[r'$\mu$']].to_numpy()
    par = sim_df[["grid_par"]].to_numpy()
    acts = sim_df[["activity"]].to_numpy()
    mus = mus.reshape((n_par, n_mus, n_trials))
    pars = par.reshape((n_par, n_mus, n_trials))
    acts = acts.reshape((n_par, n_mus, n_trials))

    f, ax = plt.subplots(1, 1)
    ax.contourf(mus[:, :, 0], pars[:, :, 0], np.mean(acts, axis=2))
    ax.axvline(neuralnetsim.calc_mu(original_graph, "level1"))
    f.savefig(save_dir.joinpath(prefix + "_act_contour.pdf"))
    plt.close()
    plt.clf()
