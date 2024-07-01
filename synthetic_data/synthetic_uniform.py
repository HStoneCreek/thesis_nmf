import numpy as np
from scipy.stats import poisson

def generate_synthetic_data(
        U: int = 90,
        K: int = 3,
        M: int = 300,
        seed: int = 42):
    """
    Generate synthetic data using a Poisson generative model with specified parameters.

    Parameters:
        U (int): Number of patients.
        K (int): Number of latent features.
        M (int): Number of observed markers.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - y (numpy.ndarray): The generated data matrix (U x M), sampled from a Poisson distribution.
            - true_classes (numpy.ndarray): An array indicating the true class for each patient.
            - lambda_matrix (numpy.ndarray): The matrix of Poisson intensities (U x M).
    """
    # Set the random seed for numpy's random number generator
    np.random.seed(seed)

    # Initialize theta with random values between 0.1 and 0.2 for each user and class
    theta = np.random.uniform(0.1, 0.2, size=(U, K))

    # Define true classes for U users, evenly distributing among K classes
    true_classes = np.concatenate([np.repeat(k, U // K) for k in range(K)])

    # Adjust theta values to emphasize differences among classes by adding random values
    for k in range(K):
        start = k * U // K
        end = (k + 1) * U // K
        theta[start:end, k] += np.random.uniform(1, 1.5, size=(U // K))


    # Initialize beta with random values, adjusting for latent features' effect on M variables
    beta = np.random.uniform(0.8, 1, size=(K, M))

    for k in range(K):
        start = k * M // K
        end = (k + 1) * M // K
        beta[k, start:end] += np.random.uniform(0.5, 1, size=(M // K))

    # Compute the matrix of Poisson intensities as the product of theta and beta
    lambda_matrix = np.dot(theta, beta)

    # Sample the observed data y from the Poisson distribution based on the computed intensities
    y = np.zeros((U, M), dtype=int)
    for u in range(U):
        for i in range(M):
            y[u, i] = poisson.rvs(mu=lambda_matrix[u, i], size=1)
    print("Mean: ",np.mean(y))
    print("STD: ", np.std(y))
    # Generate a binomial mask with a 30% chance

    # Generate Poisson random variables
    poisson_values = poisson.rvs(mu=0.5, size=(U, M))

    # Create a random sign array with +1 or -1 (50% chance each)
    signs = np.random.choice([1, -1], size=(U, M))

    # Apply the mask, and either add or subtract the Poisson values based on the sign
    modification = signs * poisson_values

    print("Mean noise: ",np.mean(modification))

    # Modify y
    y += modification
    print(np.mean(y))
    # Ensure all values in y are non-negative
    y = np.maximum(y, 0)
    # Print mean values for debug purposes
    print("MEAN THETA: ", np.mean(theta), "MEAN BETA: ", np.mean(beta), "MEAN y: ", np.mean(y))

    return y, true_classes, lambda_matrix


