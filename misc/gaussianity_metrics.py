import numpy as np

def compute_kurtosis(samples):
    mean = np.mean(samples, axis=0, keepdims=True)
    std = np.std(samples, axis=0, keepdims=True) + 1e-12
    z = (samples - mean) / std

    kurt = np.mean(z**4, axis=0) - 3.0  # excess kurtosis
    return float(np.mean(np.abs(kurt)))  # Note: we calculate the ABSOLUTE VALUE of the excessive kurtosis.


def kl_to_gaussian_1d(samples, n_bins=50):
    # samples: (n_samples, N)
    kl_vals = []

    for i in range(samples.shape[1]):
        x = samples[:, i]

        mu = np.mean(x)
        sigma = np.std(x) + 1e-12

        hist, bin_edges = np.histogram(x, bins=n_bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        p = hist + 1e-12

        q = (1.0 / (np.sqrt(2*np.pi)*sigma)) * np.exp(-0.5 * ((bin_centers - mu)/sigma)**2)
        q = q + 1e-12

        kl = np.sum(p * np.log(p / q)) * (bin_edges[1] - bin_edges[0])
        kl_vals.append(kl)
    return float(np.mean(kl_vals))