from dask.distributed import Client, LocalCluster
import neuralnetsim
import numpy as np
import pandas
import seaborn
import matplotlib.pyplot as plt
np.seterr(all='raise')


def sphere(x, data):
    return sum(x[i]**2 for i in range(len(x)))


if __name__ == "__main__":
    cluster = LocalCluster(n_workers=5, threads_per_worker=1)
    client = Client(cluster)
    es = neuralnetsim.SCMAEvoStrat(x0=np.array([0.5, 1.2, 3.5, 0.01]),
                                   sigma0=2.4,
                                   seed=34)
    es.run(sphere, client, 50, cost_kwargs={"data": {"inside": True}},
           enable_path_history=True)
    cost_data = pandas.DataFrame(
        {
            'cost': cost,
            'generation': es.generation_history[i]
        }
        for i in range(len(es.cost_history)) for cost in es.cost_history[i]
    )
    data = pandas.DataFrame(
        {
            'sigma': es.sigma_history[i],
            'sigma_path': es.sigma_path_history[i],
            'cov_path': es.cov_path_history[i],
            'centroid': np.mean(es.centroid_history[i]),
            'generation': es.generation_history[i]
        }
        for i in range(len(es.cost_history))
    )
    f, axes = plt.subplots(nrows=5, sharex=True)
    seaborn.lineplot(data=cost_data, x="generation", y="cost", ax=axes[0])
    seaborn.lineplot(data=data, x="generation", y="centroid", ax=axes[1])
    seaborn.lineplot(data=data, x="generation", y="sigma", ax=axes[2])
    seaborn.lineplot(data=data, x="generation", y="sigma_path", ax=axes[3])
    seaborn.lineplot(data=data, x="generation", y="cov_path", ax=axes[4])
    plt.tight_layout()
    plt.savefig("scmaes_example.pdf")
    plt.close()
    plt.clf()
