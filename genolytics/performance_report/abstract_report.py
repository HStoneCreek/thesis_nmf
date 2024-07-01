class AbstractReport:

    def __init__(self,
                 execution_time: float,
                 retained_components: int,
                 rand_score: float,
                 no_pred_clusters: int,
                 true_clusters: int,
                 accuracy: float,
                 entropy: float,
                 purity: float,
                 largest_cluster: float,
                 sparsity: float
                ):
        self.execution_time = execution_time
        self.retained_components = retained_components
        self.rand_score = rand_score
        self.no_pred_clusters = no_pred_clusters
        self.true_clusters = true_clusters
        self.accuracy = accuracy
        self.entropy = entropy
        self.purity = purity
        self.largest_cluster = largest_cluster
        self.sparsity = sparsity
