import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from genolytics.executions.load_data import load_barley

from genolytics.utils import compute_best_accuracy, majority_class_accuracy, compute_purity, compute_entropy, hinton

config = {
    3: [3456,23456, 34567,  45678, 56789],
    10: [1234, 2345, 3456, 4567, 5678],
    6: [2345, 4567, 3456, 2345]
}

K = 6
DROP_OUT = 2
HINTON = False
OPTIMAL_K = False
AGGREGATE = False
WORLD_MAP = False
seeds = config.get(K)
acc, ra, pu, en = [], [], [], []
X, y = load_barley(drop_meta=AGGREGATE)
for seed in seeds:
    try:
        df = pd.read_csv(f"r_barley/barley_dropout_seed_{seed}_K_{K}_dropout_{DROP_OUT}.csv", index_col=0)
    except:
        continue

    if DROP_OUT <1:
        keep = pd.read_csv(f"r_barley/permutation_barley_seed_{seed}_K_{K}_dropout_{DROP_OUT}.csv", index_col=0)
        y = y[keep["x"]-1,]

    kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=500, n_init=10, random_state=43)
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
               y_labels=list("ABCDEF"),
               class_labels=y)
    if OPTIMAL_K:
        # Range of k values to try
        k_values = range(2, 11)

        # Dictionary to hold the Silhouette scores for different values of k
        silhouette_scores = {}

        # Apply k-Means for each k, compute the Silhouette Score
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++', max_iter=500, n_init=10)
            cluster_labels = kmeans.fit_predict(X)

            # Calculate the silhouette scores
            score = silhouette_score(X, cluster_labels)
            silhouette_scores[k] = score
            print(f'Silhouette Score for k={k}: {score}')

        # Find the key with the highest Silhouette Score
        best_k = max(silhouette_scores, key=silhouette_scores.get)
        print(f'Best k by Silhouette Score: {best_k} with a score of {silhouette_scores[best_k]}')

    if not AGGREGATE:
        X["clusters"] = clusters

        # Creating a new column for easier pivoting (combining all classifications)
        X['Classification'] = X['Row_type'].astype(str) + " | " + X['Breeding History'] + " | " + X['Growth habit']

        # Pivot table to count occurrences of cluster assignments within each classification
        pivot_table = pd.pivot_table(X[["clusters", "Classification"]], index='Classification', columns='clusters', aggfunc=len,
                                     fill_value=0)

        # Adding row totals
        pivot_table['Total'] = pivot_table.sum(axis=1)

        # Adding column totals
        pivot_table.loc['Total', :] = pivot_table.sum()
        print(pivot_table)

    if WORLD_MAP:

        import pycountry

        country_corrections = {
            "USA": "United States",
            "Russia": "Russian Federation",
            "Bolivia": "Bolivia, Plurinational State of",
            "Iran": "Iran, Islamic Republic of",
            "Venezuela": "Venezuela, Bolivarian Republic of",
            "South Korea": "Korea, Republic of",
            "North Korea": "Korea, Democratic People's Republic of",
            "Germany/Czech Republic": "Germany",  # Take the first mentioned
            "Germany/Netherlands": "Germany",  # Take the first mentioned
            "Ex. Yugoslavia": "Yugoslavia",  # Assuming you want former Yugoslavia
            "Republic of Korea": "Korea, Republic of",
            "Finland/Sweden": "Finland",  # Take the first mentioned
            "Yugoslavia": "Serbia",  # Assuming modern equivalent, Serbia was part of former Yugoslavia
            "Tibet": "China"
        }

        def get_iso_code(country_name):
            try:
                return pycountry.countries.lookup(country_name).alpha_3
            except LookupError:
                return None


        X['iso_code'] = X["Country"].apply(get_iso_code)

        X['Country'] = X['Country'].apply(lambda x: country_corrections.get(x, x))

        X['iso_code'] = X["Country"].apply(get_iso_code)

        import geopandas as gpd

        # Load the built-in GeoDataFrame for world map
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

        world = world.merge(X[['iso_code', 'clusters']], how='left', left_on='iso_a3', right_on='iso_code')
        import matplotlib.pyplot as plt

        world['clusters'] = pd.Categorical(world['clusters'])  # Example categorical data
        # Create a color map
        num_clusters = len(world['clusters'].unique())
        colormap = plt.get_cmap('viridis', num_clusters)  # Updated usage

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        world.plot(column='clusters', cmap=colormap, ax=ax, legend=False)

        # Assuming 'colormap' is already created and 'clusters' is a column in 'world'
        legend_handles = [Patch(color=colormap(i / num_clusters), label=f'Cluster {i + 1}') for i in
                          range(num_clusters)]
        ax.legend(handles=legend_handles, title="Clusters")
        plt.show()



print("Acc: ",sum(acc)/ len(seeds))
print("Purity: ",sum(pu)/ len(seeds))
print("Entropy: ",sum(en)/ len(seeds))
print("Rand: ",sum(ra)/ len(seeds))




