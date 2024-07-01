from time import time
from typing import Dict, Optional
import matplotlib.pyplot as plt

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans

from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from genolytics.executions.load_data import load_leukemia_data
from genolytics.models.abstract import AbstractModel
from genolytics.performance_report.nmf_report import NMFReport
from genolytics.utils import logger, compute_best_accuracy, majority_class_accuracy, compute_purity, compute_entropy, \
    hinton


class NMFModel(AbstractModel):

    def __init__(self,
                 X: np.ndarray,
                 y: Optional[np.ndarray] = None,
                 dataset_name: str = '',
                 seed: int = 42):
        super().__init__(X, y, dataset_name, seed)
        self.X = X
        self.y = y
        self.dataset_name = dataset_name

        self.seed = seed
        self._parameter = {}
        self.W: Optional[np.ndarray] = None
        self.H: Optional[np.ndarray] = None
        self.solver = None
        self.beta_loss = None
        self.clusters = None
        self.n_components = None
        self.execution_time: float = 0
        self.l1_ratio: float = 0
        self.alpha_W: float = 0
        self.alpha_H: float = 0

        logger.info(f"Initialized {self.model_name} using {self.dataset_name} data.")

    @property
    def model_name(self) -> str:
        penalty = ""
        prefix = ""
        if self.l1_ratio == 0 and self.penalty is not None:
            penalty = " (L2)"
            prefix = "sparse "
        elif self.l1_ratio == 1 and self.penalty is not None:
            penalty = " (L1)"
            prefix = "sparse "
        elif self.l1_ratio == 0.5 and self.penalty is not None:
            penalty = " (Elastic Net)"
            prefix = "sparse "

        name = f"{prefix}NMF{penalty} lambda={self.penalty}" if self.penalty is not None else "NMF"
        return name

    @property
    def parameter(self):
        return self._parameter

    @parameter.setter
    def parameter(self, value):
        self._parameter = dict(value)

    def preprocess(self):
        pass

    def reduce(self,
               n_components: int,
               alpha_W: float = 0,
               alpha_H: float = "same",
               l1_ratio: float = 0,
               solver: str = "cd",
               beta_loss: str = "frobenius",
               *args, **kwargs):

        self.n_components = n_components
        self.l1_ratio = l1_ratio
        self.alpha_H = alpha_H
        self.alpha_W = alpha_W
        self.solver = solver
        self.beta_loss = beta_loss
        print(f"Start NMF model fit - alpha H: {self.alpha_H}, alpha W: {self.alpha_W} l1 ratio: {l1_ratio}, "
              f"solver: {self.solver}, beta loss: {self.beta_loss}, n_components: {self.n_components}")

        self.penalty = self.alpha_W if self.alpha_W != 0 or self.alpha_H != 0 else None

        model3 = NMF(n_components=self.n_components,
                     beta_loss=self.beta_loss,
                     solver=self.solver,
                     random_state=self.seed,
                     alpha_W=alpha_W,
                     alpha_H=alpha_H,
                     l1_ratio=l1_ratio,
                     max_iter=20_000,
                     init='nndsvda',
                     tol=1e-5
                     )
        self.W = model3.fit_transform(self.X)
        self.H = model3.components_
        return self

    def cluster(self,
                linkage_method="average",
                method: str = "KMEANS",
                show_plot: bool = True,
                n_clusters: int = 2,
                *args, **kwargs):
        if method == "NMF":
            self.clusters = np.argmax(self.W, axis=1) + 1

            Z = linkage(self.W, method=linkage_method)

            if show_plot:
                plt.figure(figsize=(20, 6))
                dendrogram(Z)

                plt.title(f'HC on {self.dataset_name} dataset with seed: {self.seed}. \n Parameters: {self.parameter}')
                plt.xlabel('Sample index')
                plt.ylabel('Distance')
                plt.show()
        else:
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=500, n_init=10, random_state=self.seed)
            kmeans.fit(self.W)

            # Predict the cluster for each data point
            self.clusters = kmeans.predict(self.W)

    def evaluate(self):

        # Assuming 'W' is the matrix obtained from NMF
        zero_count_W, zero_count_H = np.sum(self.W == 0), np.sum(self.H == 0)
        total_elements_W, total_elements_H = self.W.size, self.H.size
        # Calculate sparsity
        sparsity_W = zero_count_W / total_elements_W
        sparsity_H = zero_count_H / total_elements_H

        logger.info(f"Sparsity W: {sparsity_W}, H: {sparsity_H}")

        if self.y is None:
            logger.info("Evaluation is not possible as y is None.")

        # Compute the Adjusted Rand Score
        rand = adjusted_rand_score(self.y, self.clusters)

        logger.info(f"Adjusted rand score is: {rand}.")

        mis = adjusted_mutual_info_score(self.y, self.clusters)
        logger.info(f"Adjusted mutual score info is: {mis}")

        best_accuracy = compute_best_accuracy(true_labels=self.y, predictions=self.clusters)
        majority_accuracy = majority_class_accuracy(self.y)
        logger.info(f"Best accuracy is: {best_accuracy}")
        logger.info(f"Majority class prediction accuracy is: {majority_accuracy}")

        purity = compute_purity(true_labels=self.y, cluster_assignments=self.clusters)
        entropy = compute_entropy(true_labels=self.y, cluster_assignments=self.clusters)
        largest = np.bincount(self.clusters).max() / self.clusters.size
        logger.info(f"Largest cluster takes: {round(largest * 100, 2)}%. Purity is: {round(purity, 2)}, "
                    f"entropy is: {round(entropy, 2)}")

        self.report = NMFReport(execution_time=self.execution_time,
                                retained_components=self.W.shape[1],
                                rand_score=rand,
                                no_pred_clusters=len(set(self.clusters)),
                                true_clusters=len(set(self.y)),
                                accuracy=best_accuracy,
                                sparsity_H=sparsity_H,
                                sparsity=sparsity_W,
                                entropy=entropy,
                                purity=purity,
                                largest_cluster=largest,
                                sparsity_W=sparsity_W,
                                alpha_H=self.alpha_H,
                                alpha_W=self.alpha_W,
                                l1_ratio=self.l1_ratio)

    def fit(self, *args, **kwargs):

        self.parameter = kwargs
        start = time()
        self.reduce(*args, **kwargs)
        self.cluster(*args, **kwargs)
        end = time()
        self.execution_time = end - start
        logger.info(f"Execution time was: {round(self.execution_time, 5)} seconds.")
        if self.dataset_name == "leukemia":
            y_labels = ["ALL-B", "ALL-T", "AML"]
        elif self.dataset_name == "barley":

            y_labels = list("ABCDEF")
        else:
            y_labels = [i for i in range(len(set(self.y)))]

        file_name = f"figs/{self.dataset_name}_theta_{self.model_name}_{self.seed}_{self.solver}_{self.beta_loss}_{self.n_components}".replace("-", "_")
        print(file_name)
        hinton(self.W,
               title=f"{self.model_name} estimated \u0398 matrix",
               xlabel="Latent variables",
               ylabel="Observations",
               name=file_name,
               x_labels=[i for i in range(self.W.shape[1])],
               y_labels=y_labels,
               class_labels=self.y)


    @property
    def report(self) -> NMFReport:
        return self._report

    @report.setter
    def report(self, value):
        self._report = value


if __name__ == "__main__":

    X, y = load_leukemia_data()

    #iris = load_iris()
    #X = iris.data
    #y = iris.target

    model = NMFModel(
        X=X,
        y=y
    )

    model.fit(n_components=3)

    model.evaluate()


