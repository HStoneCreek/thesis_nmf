import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from genolytics.executions.load_data import load_leukemia_data, load_barley
from synthetic_data.synthetic_uniform import generate_synthetic_data


def analyze_poisson_data(data, dataset_name, xlim_max):
    # Load the dataset
    data = pd.Series(data.values.reshape(1,-1)[0])
    # Generate basic statistics
    desc = data.describe()
    print(f"Descriptive Statistics of {dataset_name}:")
    print(desc)

    # Create histogram
    plt.figure()
    plt.hist(data, bins=100, alpha=0.75, range=(0, xlim_max), edgecolor='black')
    #plt.yticks(np.arange(0, 80000 + 1, 5000))
    plt.title(f'Distribution of {dataset_name} data')
    if dataset_name == 'barley':
        plt.xticks(np.arange(0, 3, 1))
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(f'histogram_{dataset_name}.png')
    plt.close()


X,y = load_leukemia_data()

analyze_poisson_data(pd.DataFrame(np.round(np.sqrt(X))), 'leukemia', xlim_max=60)


X, y, _ = generate_synthetic_data()

analyze_poisson_data(pd.DataFrame(X), 'synthetic', xlim_max=15)


X,y = load_barley()
analyze_poisson_data(pd.DataFrame(X), 'barley', xlim_max=2)


