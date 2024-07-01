from abc import abstractmethod
from time import time
from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

import matplotlib.pyplot as plt

from genolytics.models.abstract import AbstractModel
from genolytics.performance_report.pc_hierarchical_report import PCHierarchicalReport
from genolytics.utils import logger, compute_best_accuracy, majority_class_accuracy, compute_purity, compute_entropy


class AbstractPCHierarchical(AbstractModel):

    def __init__(self,
                 X,
                 y=None,
                 data_type: str = None,
                 dataset_name: str = '',
                 seed: int = 42
                 ):

        super().__init__(X=X,
                         y=y,
                         dataset_name=dataset_name,
                         seed=seed)
        self.X = X
        self.y = y
        self.data_type = data_type
        self._report: Optional[PCHierarchicalReport] = None
        self._parameter = {}

        self.principal_components: Optional[np.ndarray] = None
        self.clusters_fcluster: Optional[np.ndarray] = None
        self.clusters_manual: Optional[np.ndarray] = None
        self.Z: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None
        self.execution_time: float = 0
        self.explained_variances = {}

        logger.info(f"Initialized {self.__class__.__name__} with dataset: {self.dataset_name}")

    @property
    def parameter(self):
        return self._parameter

    @parameter.setter
    def parameter(self, value):
        self._parameter = dict(value)

    def preprocess(self):
        if self.data_type == "SNP":
            # compute column mean
            means = np.mean(self.X, axis=0)
            # subtract mean by column
            self.X = (self.X - means) / np.sqrt(means * (1 - means))

        self._preprocess()

    @abstractmethod
    def _preprocess(self):
        raise NotImplementedError("Must be implemented by child for custom processing steps.")

    @abstractmethod
    def reduce(self, *args, **kwargs):
        pass

    def cluster(self,
                n_clusters: Optional[int] = None,
                method: str = "KMEANS",
                linkage_method: str = 'average',
                manual_clustering: bool = False,
                show_plot: bool = False,
                *args,
                **kwargs
                ):

        if self.principal_components is None:
            raise ValueError("No model is fitted. Execute .fit() first.")
        if n_clusters is None:
            # Range of clusters to try
            range_of_clusters = range(2, 11)

            # List to store the average silhouette scores
            silhouette_scores = {}

            # Apply KMeans and calculate silhouette score for each number of clusters
            for n_clusters in range_of_clusters:
                self.cluster(n_clusters=n_clusters)

                silhouette_avg = silhouette_score(self.y.reshape(-1, 1), self.clusters)
                silhouette_scores[n_clusters] = silhouette_avg

            n_clusters = max(silhouette_scores, key=silhouette_scores.get)
            logger.info(
                f"Determined {n_clusters} clusters as the optimal number of clusters based on silhouette scores.")
        # perform hierarchical clustering

        if method == "HC":
            self.Z = linkage(self.principal_components, method=linkage_method)

            if show_plot:
                plt.figure(figsize=(20, 6))
                dendrogram(self.Z)

                plt.title(f'HC on {self.dataset_name} dataset with seed: {self.seed}. \n Parameters: {self.parameter}')
                plt.xlabel('Sample index')
                plt.ylabel('Distance')
                plt.show()
            if manual_clustering:
                clusters = {}
                for i in sorted(set(self.y)):
                    cluster = input(f"Cluster of cluster {i}:")
                    if cluster == "":
                        missing = [index for index in range(self.principal_components.shape[0]) if
                                   index not in clusters]
                        cluster = ",".join([str(number) for number in missing])
                    elif cluster == "None":
                        continue

                    index_pos = cluster.split(",")
                    index_pos = [int(pos) for pos in index_pos]
                    clusters.update({index: i for index in index_pos})

                self.clusters_manual = pd.Series(clusters)
                self.clusters_manual = self.clusters_manual.sort_index()
            # Cut the dendrogram and get cluster labels
            self.clusters_fcluster = fcluster(self.Z, n_clusters, criterion='maxclust')
        else:
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=self.seed)
            kmeans.fit(self.principal_components)

            # Predict the cluster for each data point
            self.clusters_fcluster = kmeans.predict(self.principal_components)
        return self

    def fit(self, *args, **kwargs):
        self.parameter = kwargs
        start = time()
        self.reduce(*args, **kwargs)
        self.cluster(*args, **kwargs)
        end = time()
        self.execution_time = end - start
        logger.info(f"Execution time was: {round(self.execution_time, 5)} seconds.")
        return self

    def evaluate(self):

        if self.principal_components.shape[1] == 1:
            explained = np.array([np.var(self.principal_components)])
        else:
            explained = np.diagonal(np.cov(self.principal_components, rowvar=False))
        total_variance = np.trace(np.cov(self.X, rowvar=False))

        explained_variance_ratio = explained / total_variance
        logger.info(f"Explained variance ratio: {explained_variance_ratio}")
        logger.info(f"Sum explained: {sum(explained_variance_ratio)}")
        self.explained_variances = {i: ratio for i, ratio in enumerate(explained_variance_ratio)}

        reconstruction = self.reconstruct()

        error = np.sqrt(np.mean((self.X-reconstruction)**2))
        logger.info(f"Reconstruction RMSE: {error}. Standard deviation across the data set: {np.sqrt(np.var(self.X))}")

        if self.y is None:
            logger.info("Evaluation is not possible as y is None.")
            return

        # Compute the Adjusted Rand Score
        rand = adjusted_rand_score(self.y, self.clusters_fcluster)

        logger.info(f"Adjusted rand score is: {rand}.")

        best_accuracy = compute_best_accuracy(true_labels=self.y, predictions=self.clusters_fcluster)
        majority_accuracy = majority_class_accuracy(self.y)
        logger.info(f"Best accuracy is: {best_accuracy}")
        logger.info(f"Majority class prediction accuracy is: {majority_accuracy}")
        sparsity = np.sum(self.principal_components == 0) / self.principal_components.size

        purity = compute_purity(true_labels=self.y, cluster_assignments=self.clusters_fcluster)
        entropy = compute_entropy(true_labels=self.y, cluster_assignments=self.clusters_fcluster)
        largest = np.bincount(self.clusters_fcluster).max()/self.clusters_fcluster.size
        logger.info(f"Largest cluster takes: {round(largest*100, 2)}%. Purity is: {round(purity, 2)}, "
                    f"entropy is: {round(entropy, 2)}")

        self._evaluate()

        self.report = PCHierarchicalReport(execution_time=self.execution_time,
                                           retained_components=self.principal_components.shape[1],
                                           explained_variance1=self.explained_variances.get(0),
                                           explained_variance2=self.explained_variances.get(1),
                                           rand_score=rand,
                                           no_pred_clusters=len(set(self.clusters_fcluster)),
                                           true_clusters=len(set(self.y)),
                                           accuracy=best_accuracy,
                                           entropy=entropy,
                                           purity=purity,
                                           largest_cluster=largest,
                                           sparsity=sparsity)

    @abstractmethod
    def _evaluate(self):
        raise NotImplementedError("Must be implemented by child for custom evaluations.")
    @abstractmethod
    def reconstruct(self):
        raise NotImplementedError("Must be implemented by child for custom recosntruction.")

    @property
    def report(self) -> PCHierarchicalReport:
        return self._report

    @report.setter
    def report(self, value):
        self._report = value

    def plot(self):
        if self.Z is None:
            raise ValueError(f"No model is fitted. Use .fit() first!")

    def dendogram_clustering(self):

        plt.figure(figsize=(20, 6))
        dendrogram(self.Z)

        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plt.show()

    def elbow(self, explained_variance_ratio_, ylim_max=0.5):

        cumulative_explained_variance = explained_variance_ratio_# np.cumsum(explained_variance_ratio_)

        # Create the elbow plot
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o',
                 linestyle='--')
        plt.title(f'Elbow Plot for PCA on {self.dataset_name} data')
        plt.xlabel('Number of Components')
        plt.ylabel('Variance explained')
        plt.grid(True)
        plt.ylim(0, ylim_max)
        plt.xticks(range(1, len(cumulative_explained_variance) + 1))  # Adding 1 because range starts at 1 not 0
        plt.tight_layout()

        plt.savefig(f"figs/elbow_pca_{self.dataset_name}_{self.seed}.png", dpi=300)
        plt.show()

