import pickle
import pandas as pd

from genolytics.benchmark.abstract_benchmark import AbstractBenchmark

from genolytics.benchmark.tables.table_runs import TableRunsPCA
from genolytics.models.abstract_pc_clustering import AbstractPCHierarchical
from genolytics.models.pc_clustering import PCClustering


class PCABenchmark(AbstractBenchmark):

    def __init__(self, model: AbstractPCHierarchical):
        super().__init__(model)


    def write_results(self,
                      dropout: float):

        W_serialized = pickle.dumps(self.model.W)
        Z_serialized = pickle.dumps(self.model.principal_components)

        report = self.model.report

        pca_result = TableRunsPCA(
            model_name=self.model.model_name,
            dataset=self.model.dataset_name,
            parameter=self.model.parameter,
            random_state=self.model.seed,
            dropout=dropout,
            alpha=self.model.parameter.get("alpha", 0),
            normalize=self.model.parameter.get("normalize"),
            linkage=self.model.parameter.get("linkage_method"),
            used_observations=self.model.X.shape[0],
            execution_time=report.execution_time,
            retained_components=report.retained_components,
            explained_variance1=report.explained_variance1,
            explained_variance2=report.explained_variance2,
            rand_score=report.rand_score,
            accuracy=report.accuracy,
            sparsity=report.sparsity,
            largest_cluster=report.largest_cluster,
            entropy=report.entropy,
            purity=report.purity,
            no_pred_clusters=report.no_pred_clusters,
            true_clusters=report.true_clusters,
            Z_matrix=Z_serialized,
            W_matrix=W_serialized,
            cluster_algo=pd.Series(self.model.clusters_fcluster).to_json(),
            cluster_manual=self.model.clusters_manual.to_json() if self.model.clusters_manual is not None else {},
            cluster_true=pd.Series(self.model.y).to_json()
        )
        self.session.add(pca_result)
        self.session.commit()



if __name__ == "__main__":
    pass



