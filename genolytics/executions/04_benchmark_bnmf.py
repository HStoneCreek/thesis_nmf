import itertools

from genolytics.benchmark.nmf_benchmark import NMFBenchmark
from genolytics.data_examples.synthetic_uniform import generate_synthetic_data
from genolytics.executions.load_data import load_medulloblastoma, load_leukemia_data, load_barley, \
    load_synthetic_SNP_hardy_weinberg, load_synthetic_SNP_gamma, load_synthetic_SNP_pattern
from genolytics.models.bnmf_gamma_poisson import bNMFModel


import numpy as np

seeds = [42, 24, 2020, 3040, 1234]
#seeds = [42]

latent = [3]

combinations = list(itertools.product(seeds, latent))

X_syn, y_syn, _ = generate_synthetic_data()

X_leukemia, y_leukemia = load_leukemia_data()

X_leukemia = np.round(np.log(X_leukemia))

#X_medu, y_medu = load_medulloblastoma()

X_barley, y_barley = load_barley()

synthetic = False
leukemia = False
medulloblastoma = False
barley = True


def start_pipeline(dataset: str,
                    n_components: int,
                    n_clusters: int,
                   penalty: str
                   ):
    if dataset == "synthetic":
        X = X_syn.copy()
        y = y_syn.copy()

    elif dataset == "barley":
        X = X_barley.copy()
        y = y_barley.copy()
    else:
        raise RuntimeError(f"Dataset unknown.")
    print(X.shape)
    model = bNMFModel(
        X=X,
        y=y,
        seed=seed,
        dataset_name=dataset
    )

    bench = NMFBenchmark(model=model)

    bench.evaluate(
        dropout=None,
        method="KMEANS",
        penalty=penalty,
        n_components=n_components,
        n_clusters=n_clusters,
        iterations=10_000
    )

for seed, latent in combinations:
    print(f"Start run with seed={seed}")

    if synthetic:

        model = bNMFModel(
            X=X_syn.copy(),
            y=y_syn.copy(),
            seed=seed,
            dataset_name="synthetic"
        )

        bench = NMFBenchmark(model=model)

        bench.evaluate(
            dropout=None,
            method="KMEANS",
            n_components=latent,
            n_clusters=3,
            iterations=100_000
        )

    if leukemia:
        model = bNMFModel(
            X=X_leukemia.copy(),
            y=y_leukemia.copy(),
            seed=seed,
            dataset_name="leukemia_gamma"
        )

        bench = NMFBenchmark(model=model)

        bench.evaluate(
            dropout=None,
            method="KMEANS",
            n_components=latent,
            n_clusters=3,
            iterations=20000
        )

    # if medulloblastoma:
    #     model = bNMFModel(
    #         X=X_medu.copy(),
    #         y=y_medu.copy(),
    #         seed=seed,
    #         dataset_name="medulloblastoma"
    #     )
    #
    #     bench = NMFBenchmark(model=model)
    #
    #     bench.evaluate(
    #         init='random',
    #         method="KMEANS",
    #         beta_loss="frobenius",
    #         solver="mu",
    #         dropout=None,
    #         n_components=2,
    #         n_clusters=2,
    #         tol=1e-6,
    #         iterations=20000
    #     )
    if barley:
        # model = bNMFModel(
        #     X=X_barley.copy(),
        #     y=y_barley.copy(),
        #     seed=seed,
        #     dataset_name="barley"
        # )
        #
        # bench = NMFBenchmark(model=model)
        #
        # bench.evaluate(
        #     dropout=None,
        #     method="KMEANS",
        #     n_components=41,
        #     n_clusters=6,
        #     iterations=20000
        # )

        model = bNMFModel(
            X=X_barley.copy(),
            y=y_barley.copy(),
            seed=seed,
            dataset_name="barley"
        )

        bench = NMFBenchmark(model=model)

        bench.evaluate(
            dropout=None,
            method="KMEANS",
            n_components=6,
            n_clusters=6,
            iterations=5000
        )

    # if syn:
    #     # start_pipeline(dataset="synthetic_SNP_hardy",
    #     #                n_components=5,
    #     #                n_clusters=3,
    #     #                penalty=None)
    #     #
    #     # start_pipeline(dataset="synthetic_SNP_hardy",
    #     #                n_components=5,
    #     #                n_clusters=3,
    #     #                penalty="gamma")
    #     #
    #     # start_pipeline(dataset="synthetic_SNP_hardy",
    #     #                n_components=5,
    #     #                n_clusters=3,
    #     #                penalty="beta")
    #
    #     # start_pipeline(dataset="synthetic_SNP_hardy",
    #     #                n_components=5,
    #     #                n_clusters=3,
    #     #                penalty="gamma_prec")
    #
    #     # start_pipeline(dataset="synthetic_SNP_gamma",
    #     #                n_components=5,
    #     #                n_clusters=3,
    #     #                penalty=None)
    #     #
    #     # start_pipeline(dataset="synthetic_SNP_gamma",
    #     #                n_components=5,
    #     #                n_clusters=3,
    #     #                penalty="gamma")
    #     #
    #     # start_pipeline(dataset="synthetic_SNP_gamma",
    #     #                n_components=5,
    #     #                n_clusters=3,
    #     #                penalty="beta")
    #
    #     start_pipeline(dataset="synthetic_SNP_gamma",
    #                    n_components=5,
    #                    n_clusters=3,
    #                    penalty="gamma_prec")
    #
    #     # start_pipeline(dataset="synthetic_SNP_pattern",
    #     #                n_components=5,
    #     #                n_clusters=3,
    #     #                penalty=None)
    #     #
    #     # start_pipeline(dataset="synthetic_SNP_pattern",
    #     #                n_components=5,
    #     #                n_clusters=3,
    #     #                penalty="gamma")
    #     #
    #     # start_pipeline(dataset="synthetic_SNP_pattern",
    #     #                n_components=5,
    #     #                n_clusters=3,
    #     #                penalty="beta")
    #     #
    #     # start_pipeline(dataset="synthetic_SNP_pattern",
    #     #                n_components=5,
    #     #                n_clusters=3,
    #     #                penalty="gamma_prec")
    #
    #     # start_pipeline(dataset="barley",
    #     #                n_components=6,
    #     #                n_clusters=6,
    #     #                penalty=None)
    #     #
    #     # start_pipeline(dataset="barley",
    #     #                n_components=6,
    #     #                n_clusters=6,
    #     #                penalty="gamma")
    #     #
    #     # start_pipeline(dataset="barley",
    #     #                n_components=6,
    #     #                n_clusters=6,
    #     #                penalty="beta")
    #     #
    #     # start_pipeline(dataset="barley",
    #     #                n_components=6,
    #     #                n_clusters=6,
    #     #                penalty="gamma_prec")
