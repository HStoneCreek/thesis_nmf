from sqlalchemy import create_engine

import pandas as pd

pd.set_option('display.max_columns', None)
# Create an engine to connect to the SQLite database
engine = create_engine('sqlite:///pca_results.db')


# Define your SQL query
query = "SELECT dataset, model_name, normalize, retained_components, alpha, rand_score, accuracy, purity, entropy, largest_cluster, sparsity, dropout FROM benchmark_runs_pca"

df = pd.read_sql_query(query, engine.raw_connection())
df["dropout"] = df["dropout"].fillna(0)
# [df["dataset"] == "leukemia"]
df_avg = df.groupby(["dataset",
                     "model_name",
                     "normalize", "alpha", "retained_components", "dropout"])[['rand_score',
                                    'accuracy',
                                    'purity',
                                    'entropy',
                                    'largest_cluster',
                                    'sparsity'
                                    ]].mean().reset_index()

print(df_avg.sort_values(["dataset", "model_name"]))

