import pickle

from genolytics.benchmark.abstract_benchmark import AbstractBenchmark

from genolytics.benchmark.tables.table_runs import TableRunsNMF
from genolytics.models.nmf import NMFModel
from genolytics.performance_report.nmf_report import NMFReport


class NMFBenchmark(AbstractBenchmark):

    def __init__(self, model: NMFModel):
        super().__init__(model=model)

    def write_results(self,
                      dropout: float):
        W_serialized = pickle.dumps(self.model.W)

        report: NMFReport = self.model.report

        pca_result = TableRunsNMF(
            model_name=self.model.model_name,
            dataset=self.model.dataset_name,
            parameter=self.model.parameter,
            random_state=self.model.seed,
            dropout=dropout,
            penalty=self.model.penalty,
            used_observations=self.model.X.shape[0],
            execution_time=report.execution_time,
            retained_components=report.retained_components,
            rand_score=report.rand_score,
            accuracy=report.accuracy,
            sparsity_H=report.sparsity_H,
            l1_ratio=report.l1_ratio,
            alpha_H=report.alpha_H,
            alpha_W=report.alpha_W,
            purity=report.purity,
            sparsity=report.sparsity,
            entropy=report.entropy,
            largest_cluster=report.largest_cluster,
            no_pred_clusters=report.no_pred_clusters,
            true_clusters=report.true_clusters,
            W_matrix=W_serialized,
            cluster_assignments=self.model.clusters,
            solver=self.model.solver,
            beta_loss=self.model.beta_loss
        )

        self.session.add(pca_result)
        self.session.commit()


if __name__ == "__main__":
    pass
