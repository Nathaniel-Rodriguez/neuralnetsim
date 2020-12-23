import argparse
import neuralnetsim
import numpy as np
from distributed import Client, LocalCluster
from pathlib import Path
import nest


nest.set_verbosity(40)


def main():
    # CLI
    parser = argparse.ArgumentParser(description='Fit to data.')
    parser.add_argument('--name',
                        type=str,
                        help="name of data")
    parser.add_argument('--graphdir',
                        type=str,
                        help="location of graph directory")
    parser.add_argument('--datadir',
                        type=str,
                        help="location of data directory")
    parser.add_argument('--seed',
                        type=int,
                        help="seed for fit")
    parser.add_argument('--workers',
                        type=int,
                        help="number of workers")
    parser.add_argument('--niter',
                        type=int,
                        help="num of iterations for EA")
    args = parser.parse_args()

    # Parameter configuration
    static_neuron_parameters = {
        "V_th": 30.0
    }
    static_synaptic_parameters = {
        'model': 'tsodyks2_synapse'
    }
    static_noise_parameters = {
        'dt': 1.0
    }
    static_global_parameters = {
        # no static
    }
    translators = [
        neuralnetsim.DistributionTranslator(
            'a', "beta",
            {'a': (0.0001, 100),
             'b': (0.0001, 100)}, 0.000001, 0.4),
        neuralnetsim.DistributionTranslator(
            'b', "beta",
            {'a': (0.0001, 100),
             'b': (0.0001, 100)}, 0.000001, 0.5),
        neuralnetsim.DistributionTranslator(
            'c', "beta",
            {'a': (0.01, 100),
             'b': (0.01, 100)}, -70.0, 55.0),
        neuralnetsim.DistributionTranslator(
            'd', "beta",
            {'a': (0.01, 100),
             'b': (0.01, 100)}, 0.0001, 30.0),
        neuralnetsim.DistributionTranslator(
            'delay', "beta",
            {'a': (0.01, 100),
             'b': (0.01, 100)}, 0.2, 80.0),
        neuralnetsim.DistributionTranslator(
            'delay', "beta",
            {'a': (0.01, 100),
             'b': (0.01, 100)}, 0.2, 8.0),
        neuralnetsim.DistributionTranslator(
            'U', "beta",
            {'a': (0.01, 100),
             'b': (0.01, 100)}),
        neuralnetsim.DistributionTranslator(
            'u', "beta",
            {'a': (0.01, 100),
             'b': (0.01, 100)}),
        neuralnetsim.DistributionTranslator(
            'x', "beta",
            {'a': (0.01, 100),
             'b': (0.01, 100)},
            0.0, 10.),
        neuralnetsim.DistributionTranslator(
            'tau_rec', "beta",
            {'a': (0.01, 100),
             'b': (0.01, 100)},
            0.02, 3000.0),
        neuralnetsim.DistributionTranslator(
            'tau_fac', "beta",
            {'a': (0.01, 100),
             'b': (0.01, 100)},
            0.02, 2000.0),
        neuralnetsim.ValueTranslator('mean', 0.0, 20.0),
        neuralnetsim.ValueTranslator('std', 0.001, 10.0, True),
        neuralnetsim.ValueTranslator('weight_scale', 5e5, 5e7, True)
    ]

    cluster = LocalCluster(n_workers=args.workers, threads_per_worker=1,
                           memory_limit=15e9)
    with Client(cluster) as client:
        # load in data
        graph = neuralnetsim.load(
            Path(args.graphdir).joinpath(args.name + "_graph.pyobj"))
        data = neuralnetsim.load_spike_times(
            Path(args.datadir).joinpath(args.name).joinpath("data.mat")
        )

        # configure model
        parameters = neuralnetsim.DistributionParameters(
            graph, "izhikevich",
            translators,
            ["weight_scale"], ["mean", "std"],
            ["U", "u", "x", "tau_rec", "tau_fac", "delay"],
            ["a", "b", "c", "d"],
            static_neuron_parameters,
            static_synaptic_parameters, static_noise_parameters,
            static_global_parameters
        )
        rng = np.random.RandomState(args.seed + 87698)
        x0 = rng.random(parameters.required_array_size())

        # prepare data for fit
        data = {node: data[node]
                for node in graph.nodes()}
        xvalidation_manager = neuralnetsim.DataManager(
            data, num_folds=1, test_ratio=0.1, start_buffer=0.1)
        ava_times = neuralnetsim.avalanches_from_zero_activity(
            xvalidation_manager.get_training_fold(0),
            0.0,
            xvalidation_manager.get_duration("training", 0, 20.0)
        )[0]
        ava_durations = ava_times[:, 1] - ava_times[:, 0]

        # prepare optimizer
        es = neuralnetsim.SCMAEvoStrat(x0=x0, sigma0=0.2, seed=args.seed,
                                       population_size=args.workers)
        es.run(neuralnetsim.duration_cost, client,
               num_iterations=args.niter,  # 6hr for 1k
               cost_kwargs={
                   "circuit_parameters": parameters,
                   "kernel_parameters": {"resolution": 0.2},
                   "data_avalanche_durations": ava_durations,
                   "kernel_seeder": np.random.RandomState(args.seed),
                   "duration": 100000.0,
                   "circuit_choice": neuralnetsim.DistributionCircuit
               },
               enable_path_history=True,
               enable_rng_seed='rng')
        es.to_file(Path.cwd().joinpath(args.name + "_circuit_fit.pyobj"))
        neuralnetsim.save(parameters,
                          Path.cwd().joinpath(args.name + "_parameters.pyobj"))


if __name__ == "__main__":
    main()
