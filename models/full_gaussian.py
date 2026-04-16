import numpy as np
from numpy.linalg import eigh, slogdet, inv

def fit_full_gaussian(train_samples, eps=1e-6):
    mu = train_samples.mean(axis=0)
    X = train_samples - mu
    cov = np.cov(X, rowvar=False)
    cov = cov + eps * np.eye(cov.shape[0])
    return mu, cov

def sample_full_gaussian(n_samples, mu, cov, seed=0):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean=mu, cov=cov, size=n_samples)

def gaussian_nll(samples, mu, cov):
    d = samples.shape[1]
    X = samples - mu
    sign, logdet = slogdet(cov)
    if sign <= 0:
        return np.inf
    cov_inv = inv(cov)
    quad = np.sum((X @ cov_inv) * X, axis=1)
    return float(0.5 * np.mean(quad + logdet + d * np.log(2 * np.pi)))