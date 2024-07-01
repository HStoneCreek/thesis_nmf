import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from genolytics.executions.load_data import load_leukemia_data


from genolytics.utils import compute_best_accuracy, majority_class_accuracy, compute_purity, compute_entropy, hinton

config = {
    #3: [2345,23456, 34567,  45678, 56789],
    3: [2345, 3456, 4567, 5678, 6789, 2123],
    10: [1234, 2345, 3456, 4567, 5678],
    7: [6789, 4567, 3456],
    6: [2345, 3456, 4567, 5678, 6789, 2123]
}

K = 3
DROP_OUT = .1
HINTON = False
seeds = config.get(K)
acc, ra, pu, en = [], [], [], []
for seed in seeds:
    try:
        df = pd.read_csv(f"r_leukemia/leukemia_dropout_seed_{seed}_K_{K}_dropout_{DROP_OUT}.csv", index_col=0)
    except:
        continue

    X, y = load_leukemia_data()
    if DROP_OUT <1:
        keep = pd.read_csv(f"r_leukemia/permutation_leukemia_seed_{seed}_K_{K}_dropout_{DROP_OUT}.csv", index_col=0)
        y = y[keep["x"]-1,]

    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=500, n_init=10, random_state=43)
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
        hinton(df,
               title=f"bNMF estimated \u0398 matrix",
               xlabel="Latent variables",
               ylabel="Observations",
               name=f"figs/leukemia_theta_bNMF_{seed}_{K}",
               x_labels=[i for i in range(df.shape[1])],
               y_labels=["ALL-B", "ALL-T", "AML"],
               class_labels=y)

print("Acc: ",sum(acc)/ len(seeds))
print("Purity: ",sum(pu)/ len(seeds))
print("Entropy: ",sum(en)/ len(seeds))
print("Rand: ",sum(ra)/ len(seeds))




