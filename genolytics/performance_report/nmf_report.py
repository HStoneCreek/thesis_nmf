from genolytics.performance_report.abstract_report import AbstractReport
class NMFReport(AbstractReport):


    def __init__(self,
                 execution_time: float,
                 retained_components: int,
                 rand_score: float,
                 no_pred_clusters: int,
                 true_clusters: int,
                 accuracy: float,
                 sparsity_W: float,
                 sparsity_H: float,
                 alpha_W: float,
                 alpha_H: float,
                 l1_ratio: float,
                 entropy: float,
                 purity: float,
                 largest_cluster: float,
                 sparsity: float
                 ):
        super().__init__(execution_time,
                         retained_components,
                         rand_score,
                         no_pred_clusters,
                         true_clusters,
                         accuracy=accuracy,
                         entropy=entropy,
                         purity=purity,
                         largest_cluster=largest_cluster,
                         sparsity=sparsity
                         )

        self.sparsity_W = sparsity_W
        self.sparsity_H = sparsity_H
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio

