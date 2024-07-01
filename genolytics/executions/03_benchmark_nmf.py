import itertools

from genolytics.benchmark.nmf_benchmark import NMFBenchmark
from genolytics.data_examples.synthetic_uniform import generate_synthetic_data
from genolytics.executions.load_data import load_leukemia_data, load_barley
from genolytics.models.nmf import NMFModel
import numpy as np

seeds = [42, 24, 2020, 3040, 1234]
#seeds = [2020]


# Coefficients for the penalties
coefficients = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05,  0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1, 2 ,3]
#coefficients = [0.05]
# List of lambda functions, each multiplying by a coefficient
penalties = [
    lambda x, coef=coef: coef if np.std(x) != 0 else 0 for coef in coefficients
]
#penalties = []
# Adding the last function which calculates the mean
penalties.append(lambda x: np.mean(x))

#penalties = [lambda x: np.mean(x)]

l1_ratios = [10]
l1_ratios = [10, 0, 1, 0.5]

latent_spaces = [6]
solver = ["cd"]

#penalties = [lambda x: (np.std(x) / np.std(x)) - 1]
#l1_ratios = [10]
combinations = list(itertools.product(seeds, penalties, l1_ratios, latent_spaces, solver))
X_leukemia, y_leukemia = load_leukemia_data()
X_leukemia = np.round(np.sqrt(X_leukemia))

X_barley, y_barley = load_barley()

X_syn, y_syn, _ = generate_synthetic_data()

synthetic = False
leukemia = False
medulloblastoma = False
barley = True


for seed, penalty, l1_ratio, latent, solve in combinations:
    if l1_ratio == 10:
        l1_ratio = 0
        alpha_W = 0
        alpha_H = 0
        if penalty(X_leukemia) != np.mean(X_leukemia):
            continue
    else:
        alpha_W = penalty(X_syn)
        alpha_H = penalty(X_syn)
        if l1_ratio == 0.5:
            alpha_H = penalty(X_syn)
    if synthetic:

        print(f"Start run with seed={seed}, l1_ratio={l1_ratio}, penalty={penalty(X_syn)}")

        model = NMFModel(
            X=X_syn.copy(),
            y=y_syn.copy(),
            seed=seed,
            dataset_name="synthetic"
        )

        bench = NMFBenchmark(model=model)

        bench.evaluate(
            dropout=None,
            method="KMEANS",
            solver=solve,
            #beta_loss="kullback-leibler",
            n_components=latent,
            n_clusters=3,
            l1_ratio=l1_ratio,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            max_iter=2000
        )

    if leukemia:
        model = NMFModel(
            X=X_leukemia.copy(),
            y=y_leukemia.copy(),
            seed=seed,
            dataset_name="leukemia"
        )

        bench = NMFBenchmark(model=model)

        bench.evaluate(
            dropout=None,
            method="KMEANS",
            solver=solve,
            #beta_loss="kullback-leibler",
            n_components=latent,
            n_clusters=3,
            l1_ratio=l1_ratio,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            max_iter=2000
        )

    # if medulloblastoma:
    #     model = NMFModel(
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
    #         l1_ratio=l1_ratio,
    #         alpha_W=alpha_W,
    #         alpha_H=alpha_H,
    #         tol=1e-6,
    #         max_iter=2000
    #     )

    if barley:

        model = NMFModel(
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
            l1_ratio=l1_ratio,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            max_iter=2000
        )

    # if syn:
    #     model = NMFModel(
    #         X=X_syn.copy(),
    #         y=y_syn.copy(),
    #         seed=seed,
    #         dataset_name="synthetic"
    #     )
    #
    #     bench = NMFBenchmark(model=model)
    #
    #     bench.evaluate(
    #         dropout=None,
    #         penalty="gamma",
    #         method="KMEANS",
    #         n_components=5,
    #         n_clusters=3,
    #         l1_ratio=l1_ratio,
    #         alpha_W=alpha_W,
    #         alpha_H=alpha_H,
    #         max_iter=2000
    #     )
    #
    #     model = NMFModel(
    #         X=X_gamma.copy(),
    #         y=y_gamma.copy(),
    #         seed=seed,
    #         dataset_name="synthetic_SNP_gamma"
    #     )
    #
    #     bench = NMFBenchmark(model=model)
    #
    #     bench.evaluate(
    #         dropout=None,
    #         penalty="gamma",
    #         method="KMEANS",
    #         n_components=5,
    #         n_clusters=3,
    #         l1_ratio=l1_ratio,
    #         alpha_W=alpha_W,
    #         alpha_H=alpha_H,
    #         max_iter=2000
    #     )
    #
    #     model = NMFModel(
    #         X=X_pattern.copy(),
    #         y=y_pattern.copy(),
    #         seed=seed,
    #         dataset_name="synthetic_SNP_pattern"
    #     )
    #
    #     bench = NMFBenchmark(model=model)
    #
    #     bench.evaluate(
    #         dropout=None,
    #         penalty="gamma",
    #         method="KMEANS",
    #         n_components=5,
    #         n_clusters=3,
    #         l1_ratio=l1_ratio,
    #         alpha_W=alpha_W,
    #         alpha_H=alpha_H,
    #         max_iter=2000
    #     )
