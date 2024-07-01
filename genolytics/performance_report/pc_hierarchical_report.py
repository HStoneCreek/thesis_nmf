from genolytics.performance_report.abstract_report import AbstractReport
class PCHierarchicalReport(AbstractReport):


    def __init__(self, execution_time: float,
                 retained_components: int,
                 rand_score: float,
                 no_pred_clusters: int,
                 true_clusters: int,
                 explained_variance1: float,
                 explained_variance2: float,
                 accuracy: float,
                 purity: float,
                 entropy: float,
                 largest_cluster: float,
                 sparsity: float
                 ):
        super().__init__(execution_time, retained_components, rand_score, no_pred_clusters, true_clusters,
                         accuracy=accuracy,
                         purity=purity,
                         entropy=entropy,
                         largest_cluster=largest_cluster,
                         sparsity=sparsity)
        self.explained_variance1 = explained_variance1
        self.explained_variance2 = explained_variance2

