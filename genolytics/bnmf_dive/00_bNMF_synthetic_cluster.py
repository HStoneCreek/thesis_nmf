
import numpy as np

import pandas as pd
from sklearn.cluster import KMeans

from genolytics.data_examples.synthetic_uniform import generate_synthetic_data

from genolytics.utils import logger, compute_best_accuracy, majority_class_accuracy, compute_purity, compute_entropy, \
    hinton

df = pd.read_csv("r_synthetic/synthetic_seed_12345_K_9_dropout_212345.csv", index_col=0)

X, y, lambda_matrix = generate_synthetic_data()


kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=500, n_init=10, random_state=43)
kmeans.fit(df)

# Predict the cluster for each data point
clusters = kmeans.predict(df)

best_accuracy = compute_best_accuracy(true_labels=y, predictions=clusters)
majority_accuracy = majority_class_accuracy(y)
print(f"Best accuracy is: {best_accuracy}")
print([i for i in range(df.shape[1])])


def rescale(values, min_val=0.03, max_val=1):
    """
    Rescale the absolute values of an array from min_val to max_val.

    Args:
    values (np.array): Input array.
    min_val (float): Minimum value for the scaled output.
    max_val (float): Maximum value for the scaled output.

    Returns:
    np.array: Rescaled array.
    """
    # Take absolute values and get the maximum
    abs_values = np.abs(values)
    max_abs_value = np.max(abs_values)

    # Avoid division by zero in case all values are zero
    if max_abs_value == 0:
        return np.zeros_like(values) + min_val

    # Scale from 0 to (max_val - min_val)
    scaled_values = (abs_values / max_abs_value) * (max_val - min_val)

    # Shift up by min_val
    return scaled_values + min_val
#print(df)
hinton(df.values,
        title=f"bNMF estimated \u0398 matrix",
               xlabel="Latent variables",
               ylabel="Observations",
               x_labels=[i for i in range(df.shape[1])],
               y_labels=[0,1,2],
               class_labels=y,
       name="synthetic_estimated_theta_bNMF_k_9_24"
       )




