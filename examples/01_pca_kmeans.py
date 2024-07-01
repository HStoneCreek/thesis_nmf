from examples.utils import get_data, analyze_barley_data
from genolytics.models.pc_clustering import PCClustering


config = {
    "synthetic": {
        "principal_components": 3,
        "clusters": 3,
        "dataset": "synthetic",
    },
    "leukemia": {
        "principal_components": 4,
        "clusters": 3,
        "dataset": "leukemia",
    },
    "barley": {
        "principal_components": 6,
        "dataset": "barley",
        "clusters": 6,
    }
}


PROFILE = "barley"
SEED = 42
CONFIG = config.get(PROFILE)
DATASET = CONFIG.get("dataset")
PRINCIPAL_COMPONENTS = CONFIG.get("principal_components")
K_MEANS_CLUSTERS = CONFIG.get("clusters")

X, y = get_data(name=DATASET)


model = PCClustering(
            X=X.copy(),
            y=y.copy(),
            dataset_name=DATASET,
            seed=SEED
        )

model.fit(n_components=PRINCIPAL_COMPONENTS,
          n_clusters=K_MEANS_CLUSTERS)


model.evaluate()

if DATASET == 'barley':
    analyze_barley_data(model.clusters_fcluster)

