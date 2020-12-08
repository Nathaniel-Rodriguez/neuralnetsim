__all__ = ["plot_spike_train"]


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from pathlib import Path
from typing import List
from typing import Dict


def counter():
    count = -1
    while True:
        count += 1
        yield count


def plot_spike_train(datasets: List[Dict[int, np.ndarray]],
                     start_time: float = None,
                     stop_time: float = None,
                     save_dir: Path = None,
                     prefix: str = "",
                     show=False):
    if start_time is None:
        start_time = 0.0
    if stop_time is None:
        stop_time = max(max(spikes) for dataset in datasets
                        for spikes in dataset.values() if len(spikes) > 0)
    if save_dir is None:
        save_dir = Path.cwd()

    unique_neurons = {key for dataset in datasets
                      for key in dataset.keys()}
    print("Neurons to plot:", unique_neurons)
    count = counter()
    y_map = {neuron: count.__next__() for neuron in unique_neurons}
    print("y_map", y_map)
    y_map_inv = {loc: neuron for neuron, loc in y_map.items()}
    for i, dataset in enumerate(datasets):
        for neuron in dataset.keys():
            x = dataset[neuron][np.logical_and(dataset[neuron] >= start_time,
                                               dataset[neuron] <= stop_time)]
            y = [y_map[neuron]] * len(x)
            # print("plotting...",  neuron, "with spikes", x, "at", y)
            plt.scatter(x, y, alpha=0.5, color=cm.Set1.colors[i])
    # loc, labels = plt.yticks()

    # plt.yticks(loc, [str(v) for v in y_map_inv.values()])
    plt.legend(handles=[
        mpatches.Patch(color=cm.Set1.colors[0], label='model'),
        mpatches.Patch(color=cm.Set1.colors[1], label='data')
    ])
    plt.savefig(save_dir.joinpath(prefix + "_spike_train.png"), dpi=300)
    if show:
        plt.show()
    plt.close()
    plt.clf()


# make plot of spike trains with avalanches shown in low opacity bands


if __name__ == "__main__":
    pass
