import itertools
import numpy as np

from genolytics.benchmark.pca_benchmark import PCABenchmark
from genolytics.data_examples.synthetic_uniform import generate_synthetic_data
from genolytics.executions.load_data import load_leukemia_data, load_barley
from genolytics.models.pc_clustering import PCClustering


seeds = [42, 24, 2020, 3040, 1234]

normalizes = [True]
whiten = [False]

combinations = list(itertools.product(seeds, normalizes, whiten))

X_syn, y_syn, _ = generate_synthetic_data()
X_leukemia, y_leukemia = load_leukemia_data()
X_leukemia = np.round(np.sqrt(X_leukemia))
X_barley, y_barley = load_barley()

synthetic = False
leukemia = True
medulloblastoma = False
barley = False

for seed, normalize, whiten in combinations:

    print(f"Start run with seed={seed}, linkage={normalize}, whiten={whiten}")

    if synthetic:
        model = PCClustering(
            X=X_syn.copy(),
            y=y_syn.copy(),
            dataset_name="synthetic",
            seed=seed
        )
        bench = PCABenchmark(
            model=model
        )

        bench.evaluate(
            dropout=None,
            n_components=3,
            normalize=normalize,
            n_clusters=3,
            whiten=False,
            manual_clustering=False,
            show_plot=True,
            check_components=True
        )

    if leukemia:
        model = PCClustering(
            X=X_leukemia.copy(),
            y=y_leukemia.copy(),
            dataset_name="leukemia",
            seed=seed)

        bench = PCABenchmark(
            model=model)

        bench.evaluate(
            dropout=0.1,
            n_components=4,
            normalize=normalize,
            n_clusters=3,
            manual_clustering=False,
            show_plot=True
        )

        #plt.figure(figsize=(6, 6))
        #hinton(bench.model.principal_components)
        #plt.title('Hinton Diagram Bayesian PCA with SVD')
        #plt.show()

    ############# Perform HC just on the raw data as done in the original paper #############
    # model = PCHierarchical(
    #     X=X.copy(),
    #     y=y.copy(),
    #     dataset_name="leukemia",
    #     seed=seed)
    #
    # matrix = X.copy()
    # # Calculate the variance of each column
    # variances = np.var(matrix, axis=0)
    # # Get the indices that would sort the variances array in descending order
    # indices = np.argsort(variances)[::-1]
    # # Sort the columns of the matrix according to the indices
    # sorted_matrix = matrix[:, indices]
    #
    # model.principal_components = sorted_matrix[:, :3500].copy()
    #
    # model.cluster(n_components=5,
    #               n_clusters=3,
    #               linkage_method=linkage,
    #              normalize=True,
    #               show_plot=True)
    # if medulloblastoma:
    #     model = PCClustering(
    #         X=X_medu.copy(),
    #         y=y_medu.copy(),
    #         dataset_name="medulloblastomas",
    #         seed=seed)
    #
    #     bench = PCABenchmark(
    #         model=model)
    #
    #     bench.evaluate(
    #         dropout=None,
    #         n_components=6,
    #         normalize=normalize,
    #         n_clusters=2,
    #         manual_clustering=False,
    #         show_plot=True
    #     )

    if barley:
        model = PCClustering(
            X=X_barley.copy(),
            y=y_barley.copy(),
            dataset_name="barley",
            seed=seed)

        bench = PCABenchmark(
            model=model)

        bench.evaluate(
            dropout=None,
            n_components=7,
            normalize=normalize,
            whiten=False,
            n_clusters=6,
            manual_clustering=False,
            show_plot=True
        )

        # plt.figure(figsize=(6, 6))
        # hinton(bench.model.principal_components)
        # plt.title('Hinton Diagram Bayesian PCA with SVD')
        # plt.show()

        ########## Perform HC just on the raw data

        # model = PCHierarchical(
        #     X=X_medu.copy(),
        #     y=y_medu.copy(),
        #     dataset_name="medulloblastomas",
        #     seed=seed)
        #
        # model.principal_components = X_medu.copy()
        #
        # model.cluster(
        #     n_components=5,
        #     n_clusters=2,
        #     linkage_method=linkage,
        #     normalize=True,
        #     show_plot=True
        # )
