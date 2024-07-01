from abc import abstractmethod
from typing import Dict, Optional

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from genolytics.bnmf_dive.plot import plot_heatmap
from genolytics.utils import logger
import matplotlib.pyplot as plt


class AbstractModel:

    def __init__(self,
                 X: np.ndarray,
                 y: Optional[np.ndarray] = None,
                 dataset_name: str = '',
                 seed: int = 42):
        self.X = X
        self.y = y
        self.dataset_name = dataset_name

        self.seed = seed

        self.set_seed()
        self.penalty = None

    def set_seed(self):
        np.random.seed(self.seed)
        np.random.default_rng(self.seed)

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError(f"Will return the model name")

    @property
    @abstractmethod
    def parameter(self) -> Dict:
        raise NotImplementedError(f"Will return all model specific parameters")

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def reduce(self, *args, **kwargs):
        pass

    @abstractmethod
    def cluster(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def report(self):
        pass

    def show_color_map(self):
        # # Define the colors for the colormap (red to green)
        # colors = [(1, 0, 0), (0, 1, 0)]  # R -> G
        # n_bins = 100  # Number of bins in the colormap
        #
        # # Create the colormap
        # cm = LinearSegmentedColormap.from_list("red_green_cm", colors, N=n_bins)
        #
        # # repeat the entries multiple times to improve the plot visibility
        # repeat_times = 10
        # w_wide = np.repeat(self.W, repeat_times, axis=1)
        #
        # plt.figure(figsize=(15, 6))
        # plt.imshow(w_wide.T, interpolation='nearest', cmap=cm)
        # plt.colorbar()
        # plt.title(f"{self.model_name} using {self.dataset_name} - Showing the weight matrix with {self.W.shape[0]} observations \n")
        #
        # y_ticks = np.arange(0, w_wide.T.shape[0], 10)
        # y_labels = ['Feature {}'.format(i) for i in range(len(y_ticks))]  # Replace with your custom labels
        #
        # plt.yticks(ticks=y_ticks, labels=y_labels)
        # plt.show()
        plot_heatmap(self.W/np.max(self.W), title=f"{self.model_name} estimated \u0398 matrix", xlabel="Latent space", ylabel="Patient", cmap="Reds")
