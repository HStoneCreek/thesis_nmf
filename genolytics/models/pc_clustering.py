from abc import ABC

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

from genolytics.bnmf_dive.plot import plot_heatmap
from genolytics.models.abstract_pc_clustering import AbstractPCHierarchical
from genolytics.utils import logger, hinton

np.random.seed(42)
np.random.default_rng(42)


class PCClustering(AbstractPCHierarchical, ABC):

    def __init__(self,
                 X: np.array,
                 y: np.array = None,
                 data_type: str = None,
                 dataset_name: str = '',
                 seed: int = 42
                 ):
        """

        :param X: np.array of dimension individuals x features
        """
        super().__init__(X=X,
                         y=y,
                         data_type=data_type,
                         dataset_name=dataset_name,
                         seed=seed)

        self.pca = None
        self.cumulative_variance = None

    @property
    def model_name(self) -> str:
        return "PCA"


    def _preprocess(self):
        pass

    def reduce(self,
               n_components: int,
               normalize: bool = True,
               check_components: bool = False,
               whiten: bool = False,
               *args,
               **kwargs
               ):

        # perform the PCA
        if normalize:
            scaler = StandardScaler(with_std=False)
            self.X = scaler.fit_transform(self.X)

        if check_components:

            pca = PCA(
                n_components=min(self.X.shape),
                whiten=whiten
            )
            pc = pca.fit_transform(self.X)
            total_components = len(pca.explained_variance_ratio_)
            threshold = 1.5 * 1 / total_components

            # Find the number of components to retain
            components_to_retain = np.sum(pca.explained_variance_ratio_ > threshold)
            logger.info(f"Recommended number of components to retain: {components_to_retain}")
        logger.info(f"Picked {n_components} components. Use whiten: {whiten}")
        self.pca = PCA(
            n_components=n_components,
            whiten=whiten
        )
        self.principal_components = self.pca.fit_transform(self.X)
        self.W = self.pca.components_.T

        self.cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        logger.info(f"Variance explained: {self.pca.explained_variance_ratio_}")
        logger.info(f"Cumulative variance explained: {self.cumulative_variance}")

        return self

    def _evaluate(self):
        if self.dataset_name == "leukemia":
            self.elbow(self.pca.explained_variance_ratio_, ylim_max=0.3)
            y_labels = ["ALL-B", "ALL-T", "AML"]
        elif self.dataset_name == "barley":
            self.elbow(self.pca.explained_variance_ratio_, ylim_max=0.2)
            y_labels = list("ABCDEF")
        else:
            self.elbow(self.pca.explained_variance_ratio_, ylim_max=0.1)
            y_labels = [i for i in range(len(set(self.y)))]

        hinton(self.principal_components,
               title=f"PCA score matrix",
               xlabel="Principal components",
               ylabel="Observations",
               name=f"figs/{self.dataset_name}_estimated_score_pca_{self.seed}",
               class_labels=self.y,
               x_labels=[i for i in range(self.principal_components.shape[1])],
               y_labels=y_labels,
               rescale_values=False)

    def reconstruct(self):
        return self.pca.inverse_transform(self.principal_components)



if __name__ == "__main__":

    import pandas as pd
    # X = pd.read_csv("../../leukemia/ALL_AML_data.txt", sep="\t", header=None)
    # y = pd.read_csv("../../leukemia/ALL_AML_samples.txt", sep="\t", names=["Sample"])
    # y = y.dropna()
    # y["Sample"] = y["Sample"].astype(str)
    #
    # y['Cell_Type'] = y['Sample'].apply(lambda x: 'AML' if x.startswith('AML') else (
    #     'B-cell' if 'B-cell' in x else ('T-cell' if 'T-cell' in x else 'none')))
    # y["cluster"] = pd.factorize(y['Cell_Type'])[0] + 1
    #
    # model = PCHierarchical(X=X.transpose().values,
    #                        y=y["cluster"].values)

    iris = load_iris()
    X = iris.data
    y = iris.target

    model = PCClustering(X=X,
                         y=y)

    model.preprocess()

    model.fit(n_components=3, normalize=True,
              n_clusters=3)

    model.evaluate()


