from examples.utils import get_data, plot_continent, country_corrections, get_iso_code, get_table, \
    analyze_barley_data
from genolytics.models.nmf import NMFModel

def config_nmf(penalty_type: str, penalty):
    if penalty_type is None:
        return 0, 0, 0
    elif penalty_type == 'l1':
        return 1, penalty, penalty
    elif penalty_type == 'l2':
        return 0, penalty, penalty
    elif penalty_type == 'elastic':
        return 0.5, penalty, penalty
    else:
        raise ValueError("Invalid penalty type")


config = {
    "synthetic": {
        "latent_variables": 3,
        "clusters": 3,
        "dataset": "synthetic",
        "penalty_type": None,
        "penalty": 0,
        "solver": "cd",
        "beta_loss": "frobenius"
    },
    "leukemia": {
        "latent_variables": 3,
        "clusters": 3,
        "dataset": "leukemia",
        "penalty_type": None,
        "penalty": 0,
        "solver": "cd",
        "beta_loss": "frobenius"
    },
    "leukemia_l1": {
        "latent_variables": 3,
        "clusters": 3,
        "dataset": "leukemia",
        "penalty_type": "l1",
        "penalty": 0.8,
        "solver": "cd",
        "beta_loss": "frobenius"
    },
    "leukemia_l2": {
        "latent_variables": 3,
        "clusters": 3,
        "dataset": "leukemia",
        "penalty_type": "l2",
        "penalty": 0.01,
        "solver": "cd",
        "beta_loss": "frobenius"
    },
    "leukemia_elastic": {
        "latent_variables": 3,
        "clusters": 3,
        "dataset": "leukemia",
        "penalty_type": "elastic",
        "penalty": 0.00005,
        "solver": "cd",
        "beta_loss": "frobenius"
    },
    "barley": {
        "latent_variables": 6,
        "dataset": "barley",
        "clusters": 6,
        "penalty_type": None,
        "penalty": 0,
        "solver": "cd",
        "beta_loss": "frobenius"
    }
}

PROFILE = "barley"
SEED = 42
CONFIG = config.get(PROFILE)
DATASET = CONFIG.get("dataset")
PRINCIPAL_COMPONENTS = CONFIG.get("latent_variables")
K_MEANS_CLUSTERS = CONFIG.get("clusters")
penalty = CONFIG.get("penalty")
PENALTY_TYPE = CONFIG.get("penalty_type")

# Example usage
l1_ratio, alpha_W, alpha_H = config_nmf(penalty_type=PENALTY_TYPE, penalty=penalty)

solver = CONFIG.get("solver")
beta_loss = CONFIG.get("beta_loss")

X, y = get_data(name=DATASET)

model = NMFModel(
    X=X.copy(),
    y=y.copy(),
    dataset_name=DATASET,
    seed=SEED
)

model.fit(n_components=PRINCIPAL_COMPONENTS,
          n_clusters=K_MEANS_CLUSTERS,
          alpha_W=alpha_W,
          alpha_H=alpha_H,
          l1_ratio=l1_ratio,
          solver=solver,
          beta_loss=beta_loss)

model.evaluate()



if DATASET == 'barley':
    analyze_barley_data(model.clusters)