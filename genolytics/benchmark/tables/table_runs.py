from sqlalchemy import Column, Integer, String, JSON, Float, LargeBinary, Boolean

from genolytics.benchmark.tables.table_abstract import AbstractTable


class TableRunsPCA(AbstractTable):

    __tablename__ = "benchmark_runs_pca"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String)
    dataset = Column(String)
    parameter = Column(JSON)
    random_state = Column(Integer)
    dropout = Column(Float)
    alpha = Column(Float, default=0)
    normalize = Column(Boolean)
    linkage = Column(String)
    used_observations = Column(Integer)
    execution_time = Column(Float)

    retained_components = Column(Integer)
    explained_variance1 = Column(Float)
    explained_variance2 = Column(Float)

    rand_score = Column(Float)
    accuracy = Column(Float)
    purity = Column(Float)
    entropy = Column(Float)
    sparsity = Column(Float)
    largest_cluster = Column(Float)
    no_pred_clusters = Column(Integer)
    true_clusters = Column(Integer)

    Z_matrix = Column(LargeBinary)
    W_matrix = Column(LargeBinary)
    cluster_algo = Column(JSON)
    cluster_manual = Column(JSON)
    cluster_true = Column(JSON)


class TableRunsNMF(AbstractTable):

    __tablename__ = "benchmark_runs_nmf"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String)
    penalty = Column(String)
    dataset = Column(String)
    parameter = Column(JSON)
    random_state = Column(Integer)
    dropout = Column(Float)
    used_observations = Column(Integer)
    execution_time = Column(Float)
    solver = Column(String)
    beta_loss = Column(String)

    retained_components = Column(Integer)

    rand_score = Column(Float)
    accuracy = Column(Float)
    alpha_W = Column(Float)
    alpha_H = Column(Float)
    l1_ratio = Column(Float)
    purity = Column(Float)
    entropy = Column(Float)
    largest_cluster = Column(Float)
    sparsity = Column(Float)
    sparsity_H = Column(Float)

    no_pred_clusters = Column(Integer)
    true_clusters = Column(Integer)

    W_matrix = Column(LargeBinary)
    cluster_assignments = Column(LargeBinary)
