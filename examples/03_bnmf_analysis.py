from examples.utils import get_data, analyze_barley_data
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from genolytics.utils import compute_best_accuracy, majority_class_accuracy, compute_purity, compute_entropy, hinton


CONFIGS = {
    "synthetic":
            {
                "dataset": "synthetic",
                "clusters": 3,
                "seeds": [12345],
                "drop": 2
    },
    "barley": {
        "dataset": "barley",
        "clusters": 6,
        "seeds": [2345], #, 3456, 4567, 5678, 6789, 2123,
        "drop": 2
    },
    "leukemia":
        {
            "dataset": "leukemia",
            "clusters": 3,
            "seeds": [2345],
            "drop": 0.1
        }
}
PROFILE = "synthetic"
CONFIG = CONFIGS.get(PROFILE)
K = CONFIG.get("clusters")
DROP_OUT = CONFIG.get("drop")
DATASET = CONFIG.get("dataset")
HINTON = True
seeds = CONFIG.get("seeds")
acc, ra, pu, en = [], [], [], []
for seed in seeds:
    try:
        df = pd.read_csv(f"r_{DATASET}/{DATASET}_dropout_seed_{seed}_K_{K}_dropout_{DROP_OUT}.csv", index_col=0)
    except:
        continue

    X, y = get_data(name=DATASET)
    if DROP_OUT <1:
        keep = pd.read_csv(f"r_{DATASET}/permutation_{DATASET}_seed_{seed}_K_{K}_dropout_{DROP_OUT}.csv", index_col=0)
        y = y[keep["x"]-1,]

    kmeans = KMeans(n_clusters=K, init='k-means++', max_iter=500, n_init=10, random_state=43)
    kmeans.fit(df)

    # Predict the cluster for each data point
    clusters = kmeans.predict(df)

    best_accuracy = compute_best_accuracy(true_labels=y, predictions=clusters)
    majority_accuracy = majority_class_accuracy(y)
    rand = adjusted_rand_score(y, clusters)
    purity = compute_purity(true_labels=y, cluster_assignments=clusters)
    entropy = compute_entropy(true_labels=y, cluster_assignments=clusters)
    print(f"Best accuracy is: {best_accuracy}")

    acc.append(best_accuracy)
    ra.append(rand)
    pu.append(purity)
    en.append(entropy)

    if HINTON:
        if DATASET == "leukemia":
            y_labels = ["ALL-B", "ALL-T", "AML"]
        elif DATASET == "barley":

            y_labels = list("ABCDEF")
        else:
            y_labels = [i for i in range(len(set(y)))]

        hinton(df,
               title=f"bNMF estimated \u0398 matrix",
               xlabel="Latent variables",
               ylabel="Observations",
               name=f"figs/{DATASET}_theta_bNMF_{seed}_{K}",
               x_labels=[i for i in range(df.shape[1])],
               y_labels=y_labels,
               class_labels=y)

    if DATASET == 'barley':
        analyze_barley_data(clusters)

print("Acc: ",sum(acc)/ len(seeds))
print("Purity: ",sum(pu)/ len(seeds))
print("Entropy: ",sum(en)/ len(seeds))
print("Rand: ",sum(ra)/ len(seeds))





