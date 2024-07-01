
import pandas as pd
from scipy.signal import convolve2d

from genolytics.utils import show_color_map
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def proportion_zero(data):
    num_zeros = np.count_nonzero(data == 0)
    # Calculate the proportion of zeros
    total_elements = data.size
    proportion_zeros = num_zeros / total_elements
    return proportion_zeros

def plot_observed_sampled(data, true, title, label):
    s = pd.Series(data)
    q_low = s.quantile(0.01)
    q_high = s.quantile(0.99)
    # Filter the series to retain values between the 1% and 99% quantile
    filtered_s = s[(s > q_low) & (s < q_high)]

    plt.hist(filtered_s, bins=30, alpha=0.5, label=r'$T(y_{rep})$')
    plt.axvline(x=true, color='k', linestyle='-', linewidth=2, label=r'$T(y)$')

    # Add legend
    plt.legend()

    # Add title and axis labels if desired
    plt.title(title)
    plt.xlabel(label)
    plt.ylabel('Frequency')
    plt.show()

def do_posterior_predictive_checks(model, trace, true):
    with model:
        ppc = pm.sample_posterior_predictive(trace, var_names=["R"])
    zero_proportions = [proportion_zero(ppc.posterior_predictive.R[0, i].values) for i in range(ppc.posterior_predictive.R.shape[1])]
    means = [ppc.posterior_predictive.R[0, i].values.mean() for i in
                       range(ppc.posterior_predictive.R.shape[1])]
    sds = [ppc.posterior_predictive.R[0, i].values.std() for i in
             range(ppc.posterior_predictive.R.shape[1])]

    plot_observed_sampled(zero_proportions, proportion_zero(true), 'PPC on proportion of zeros', 'Zero proportions')
    plot_observed_sampled(means, true.mean(), 'PPC on mean value', 'Means')
    plot_observed_sampled(sds, true.std(), 'PPC on standard deviations', 'Stds')

    U_samples = trace.posterior["U"]
    U_hdi = np.array([[pm.hdi(U_samples[0, :, j, i].values) for j in range(U_samples.shape[2])]
                      for i in range(U_samples.shape[3])])
    U_hdi_widths = np.zeros((5, 90))
    # Compute the width of each HDI
    for i in range(5):
        for j in range(90):
            U_hdi_widths[i, j] = U_hdi[i][j][1] - U_hdi[i][j][0]  # upper bound - lower bound

    show_color_map(U_hdi_widths.T, "HDI width", "synthetic data")
    idx_matrices = np.random.randint(0, 5000, 7)
    matrices = [true]
    matrices_sample = [ppc.posterior_predictive.R[0, i].values for i in idx_matrices]
    matrices.extend(matrices_sample)
    num_matrices = 8
    # Count the occurrences of each integer in each matrix
    counts = [pd.Series(matrix.flatten()).value_counts().sort_index() for matrix in matrices]

    # Combine counts into a single DataFrame
    df_counts = pd.concat(counts, axis=1)
    df_counts.columns = [f'Matrix_{i + 1}' for i in range(num_matrices)]
    df_counts = df_counts.fillna(0).astype(int)  # Fill missing values with 0 and ensure integers

    df = df_counts
    # Create a 2x4 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    fig.tight_layout(pad=6.0)  # Adds padding between plots

    # Loop through each subplot and plot the data
    for i, ax in enumerate(axes.flat):
        category = f'Matrix_{i + 1}'
        bars = df[category].plot(kind='bar', ax=ax)
        if i+1 == 1:
            category = "Observed data"
        else:
            category = f"PPC sample {i}"

        ax.set_title(category)
        ax.set_xlabel('Group')
        ax.set_ylabel('Frequency')

        for bar in bars.patches:
            bar_height = bar.get_height()
            ax.annotate(f'{int(bar_height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar_height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')



    plt.show()


def plot_weights(data):
    fig, axs = plt.subplots(data.shape[0], 1, figsize=(10, 15))  # One subplot per category
    categories = [f"Latent space {i}" for i in range(data.shape[0])]
    # Determine the global maximum to set a uniform x-axis across all subplots
    max_val = np.max(data)
    for i, ax in enumerate(axs):
        y_pos = np.arange(data[i].size)
        ax.barh(y_pos, data[i], align='center')
        tick_positions = np.arange(0, data[i].size, 5)  # Positions for ticks every 5 observations
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_positions)  # Labels show the actual observation number, starting from 1
        ax.set_title(categories[i])
        ax.set_xlim(0, max_val + 0.05 * max_val)  # Extend x-axis slightly beyond max_val
    plt.tight_layout()
    plt.show()

def plot_bland_altman(estimate, true):

    means = (true + estimate) / 2
    differences = true - estimate
    # Plot Bland-Altman
    plt.figure(figsize=(8, 6))
    plt.scatter(means.flatten(), differences.flatten(), alpha=0.5)
    colors = ['red', 'green', 'blue']  # Colors for each column
    labels = [f'Latent space {k}' for k in range(means.shape[1])]  # Labels for each column
    for i in range(means.shape[1]):  # Loop through columns
        plt.scatter(means[:, i], differences[:, i], color=colors[i], label=labels[i])
    plt.axhline(y=differences.flatten().mean(), color='r', linestyle='--')
    plt.axhline(y=differences.flatten().mean() + 1.96 * differences.flatten().std(), color='r', linestyle='--')
    plt.axhline(y=differences.flatten().mean() - 1.96 * differences.flatten().std(), color='r', linestyle='--')
    plt.title('Bland-Altman Plot')
    plt.xlabel('Mean of True and Estimated Values')
    plt.ylabel('Differences between True and Estimated Values')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_heatmap(error_matrix, title: str, xlabel: str, ylabel: str, cmap='coolwarm'):
    plt.figure(figsize=(10, 8))
    from matplotlib.colors import LinearSegmentedColormap

    # Define a custom colormap from white to black
    cmap = LinearSegmentedColormap.from_list('white_to_black', ['white', 'black'])

    ax = sns.heatmap(error_matrix, annot=False, cmap=cmap,
                cbar_kws={'label': 'Magnitude'}, vmin=0, vmax=1)



    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)  # Adjust tick label font size
    cbar.set_label('Magnitude', fontsize=14)


    plt.show()

def plot_scatter_true_predicted(estimate, true):

    plt.figure(figsize=(8, 6))

    colors = ['red', 'green', 'blue']  # Colors for each column
    labels = [f'Latent space {k}' for k in range(true.shape[1])]  # Labels for each column
    for i in range(true.shape[1]):  # Loop through columns
        plt.scatter(true[:, i], estimate[:, i], color=colors[i], label=labels[i])
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot of True vs. Predicted Values')
    plt.plot([true.min(), true.max()], [true.min(), true.max()], 'k--')  # Diagonal line
    plt.grid(True)
    plt.legend()
    plt.show()


def get_contingency_table(reconstruction, y):

    df = pd.DataFrame({'Reconstruction': reconstruction.ravel(), 'True': y.ravel()})
    # Create a contingency table
    contingency_table = pd.crosstab(df['Reconstruction'], df['True'])
    # Reindex to ensure all possible values (0-8) appear in both dimensions
    all_values = np.arange(np.max(df.values))  # Values from 0 to 8
    contingency_table = contingency_table.reindex(index=all_values, columns=all_values, fill_value=0)

    return contingency_table


def plot_kernel_heatmap(reconstruction, y, model, kernel_size = 5):

    kernel = np.ones((1, kernel_size)) / kernel_size
    # Perform the convolution with the padded matrix
    print(np.std(y, axis=0).shape)
    convolution_result = pd.DataFrame(((y-reconstruction)/np.std(y, axis=0))).fillna(0).replace(np.inf, 0).replace(-np.inf, 0)
    convolution_result = convolve2d(convolution_result, kernel, mode='valid')

    plt.figure(figsize=(10, 8))
    sns.heatmap(convolution_result, annot=False, cmap='coolwarm', fmt=".1f")
    plt.title(f'Heatmap of reconstruction error of {model} (averaged per patient with {kernel_size}x1 kernel)')
    plt.xlabel('Markers')
    plt.ylabel('Patients')
    plt.show()




def calculate_column_differences(matrix):
    n_columns = matrix.shape[-1]
    differences = {}
    for i in range(n_columns):
        for j in range(i + 1, n_columns):
            # Calculate differences for all observations across all samples
            differences[f"Col_{i}_minus_Col_{j}"] = matrix[..., i] - matrix[..., j]
    return differences

def prepare_data_for_plotting(differences):
    diff_stats = {}
    for key, diff in differences.items():
        # Compute summary statistics for each observation (mean, hdi)
        diff_stats[key] = az.summary(xr.DataArray(diff, dims=["chain", "draw", "observation"]),
                                      hdi_prob=0.95)
    return diff_stats



def plot_observation_differences(diff_stats, differences, num_observations):
    num_pairs = len(diff_stats.keys())
    fig, axes = plt.subplots(num_observations, num_pairs, figsize=(4 * num_pairs, 2 * num_observations), squeeze=False)

    for col_idx, (pair_name, stats) in enumerate(diff_stats.items()):
        for obs_idx in range(num_observations):
            ax = axes[obs_idx][col_idx]
            data = differences[pair_name][..., obs_idx].flatten()  # Flatten chain and draw
            sns.kdeplot(data, ax=ax, fill=True)

            # Retrieve the mean and HDI from the statistics
            mean = stats.loc[f"x[{obs_idx}]", 'mean']
            hdi_low = stats.loc[f"x[{obs_idx}]", 'hdi_2.5%']
            hdi_high = stats.loc[f"x[{obs_idx}]", 'hdi_97.5%']

            ax.axvline(mean, color='black', linestyle='--')
            ax.axvspan(hdi_low, hdi_high, alpha=0.5, color='gray')
            ax.set_title(f'{pair_name} Obs x[{obs_idx}]')
            ax.set_yticks([])
            ax.set_ylim(bottom=0)  # Adjust ylim to improve visibility
            ax.set_xlim(-1, 1)

    plt.tight_layout()
    plt.show()






