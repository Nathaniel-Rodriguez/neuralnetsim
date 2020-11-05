from dask.distributed import Client, LocalCluster
import unittest
import neuralnetsim


def fit(*args, **kwargs):
    from time import sleep
    sleep(2)
    return 1.0


class TestSCMAEvoStrat(unittest.TestCase):
    def test_dask(self):

        print("MAKING LOCALCLUSTER")
        cluster = LocalCluster(n_workers=5, threads_per_worker=1,
                               processes=False, dashboard_address=None)
        print("MAKING CLIENT")
        client = Client(cluster)
        print("BOHEK SERVICE: ", client.scheduler_info()['services'])
        es = neuralnetsim.SCMAEvoStrat()
        print("RUNNING...\n")
        es.run(fit, client, 50)
