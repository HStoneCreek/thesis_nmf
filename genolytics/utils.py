import structlog

from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mode
import matplotlib.pyplot as plt
import numpy as np

logger = structlog.getLogger()

from itertools import permutations
from sklearn.metrics import accuracy_score

from sklearn.metrics.cluster import contingency_matrix


class ClassProperty:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        val = self.fget(owner_cls)
        return val


classproperty = ClassProperty



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

def hinton(matrix, title, class_labels,  xlabel, x_labels, ylabel,y_labels, name, max_weight=None, ax=None, vertical_spacing=1, horizontal_spacing=0.1,
           rescale_values = True):
    """Draw Hinton diagram for visualizing a weight matrix with enumerated ticks."""
    if rescale_values:
        matrix = rescale(matrix)
    else:
        max_weight = np.abs(matrix).max()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if not max_weight:
        pass
        #max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()))
        #max_weight = np.abs(matrix).max()

    ax.patch.set_facecolor('gray')
    ax.set_aspect('auto')

    # Calculate spacing between boxes
    height, width = matrix.shape
    ax.set_xlim(-0.5, width - 0.5 + (width - 1) * horizontal_spacing)
    ax.set_ylim(-0.5, height - 0.5 + (height - 1) * vertical_spacing)

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        if rescale_values:
            size = abs(w)#np.sqrt(abs(w) / max_weight)
        else:
            size = np.sqrt(abs(w) / max_weight)
        # Adjust coordinates to include spacing
        rect_x = y + y * horizontal_spacing - size / 2
        rect_y = x + x * vertical_spacing - size / 2
        rect = plt.Rectangle([rect_x, rect_y], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    # Find indices where the class changes, i.e., label locations
    unique_classes = np.unique(class_labels)
    first_indices = [np.where(class_labels == uc)[0][0] for uc in unique_classes]

    # Adding tick marks
    ax.set_xticks(np.arange(width) + horizontal_spacing * np.arange(width) - 0.5)


    ax.set_yticks(first_indices)
      # Ensure fontsize is appropriate

    #ax.set_yticklabels(np.arange(height))
    #ax.set_yticklabels([class_labels[idx] for idx in first_indices])

    ax.xaxis.set_major_locator(plt.FixedLocator(np.arange(width) + horizontal_spacing * (np.arange(width) - 0.5)))
    ax.yaxis.set_major_locator(plt.FixedLocator(first_indices + vertical_spacing * np.array(first_indices) - 0.5))
    ax.set_xticklabels(x_labels, ha='center')
    ax.set_yticklabels(y_labels, fontsize=10)

    ax.invert_yaxis()

    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)

    name = f"{name}.png"
    plt.savefig(name, dpi=300)
    plt.show()

    
def compute_best_accuracy(true_labels, predictions):

    # Identify all unique labels
    unique_labels = np.unique(true_labels)

    # Generate all possible permutations of the unique labels<
    label_permutations = permutations(unique_labels)

    best_accuracy = 0

    # Iterate through each permutation
    for perm in label_permutations:
        # Create a dictionary mapping from original to permuted labels
        mapping = {original: permuted for original, permuted in zip(unique_labels, perm)}

        # Apply the mapping to the predictions
        mapped_predictions = np.array([mapping[pred] for pred in predictions])

        # Calculate the accuracy with this mapping
        accuracy = accuracy_score(true_labels, mapped_predictions)

        # Update the best accuracy if this one is higher
        if accuracy > best_accuracy:
            best_accuracy = accuracy

    return round(best_accuracy, 2)

def majority_class_accuracy(true_labels):

    mode_result = mode(true_labels).mode
    accuracy = accuracy_score(true_labels, np.array([mode_result for i in true_labels]))
    return round(accuracy, 2)


def compute_purity(true_labels, cluster_assignments):
    # Create the contingency matrix
    cont_matrix = contingency_matrix(true_labels, cluster_assignments)
    # Sum the maximum counts for each cluster (along the rows)
    max_counts = np.sum(np.amax(cont_matrix, axis=0))
    # Calculate purity
    purity = max_counts / np.sum(cont_matrix)
    return purity


def compute_entropy(true_labels, cluster_assignments):
    # Create the contingency matrix
    cont_matrix = contingency_matrix(true_labels, cluster_assignments)
    # Total number of samples
    n = np.sum(cont_matrix)
    # Number of true class labels
    l = cont_matrix.shape[0]
    # Calculate the entropy for each cluster
    entropy_sum = 0
    for cluster in cont_matrix.T:  # Transpose to iterate over clusters
        cluster_size = np.sum(cluster)
        if cluster_size > 0:
            # Calculate the entropy for each class in the cluster
            class_entropies = -np.sum((cluster / cluster_size) * np.log2(cluster / cluster_size + 1e-10))  # Adding a small value to prevent log(0)
            entropy_sum += cluster_size * class_entropies
    # Normalize the entropy
    entropy = entropy_sum / (n * np.log2(l))
    return entropy


def show_color_map(W, model_name, dataset_name):
    # Define the colors for the colormap (red to green)
    colors = [(1, 1, 1), (0, 0, 0)]  # R -> G
    n_bins = 100  # Number of bins in the colormap

    # Create the colormap
    cm = LinearSegmentedColormap.from_list("white_black_cm", colors, N=n_bins)

    # repeat the entries multiple times to improve the plot visibility
    repeat_times = 10
    w_wide = np.repeat(W, repeat_times, axis=1)

    plt.figure(figsize=(15, 6))
    plt.imshow(w_wide.T, interpolation='nearest', cmap=cm)
    plt.colorbar()
    plt.title(f"{model_name} using {dataset_name} - Showing the weight matrix with {W.shape[0]} observations \n")

    y_ticks = np.arange(0, w_wide.T.shape[0], 10)
    y_labels = ['Feature {}'.format(i) for i in range(len(y_ticks))]  # Replace with your custom labels

    plt.yticks(ticks=y_ticks, labels=y_labels)
    plt.show()

