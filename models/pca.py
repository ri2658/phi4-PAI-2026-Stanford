import numpy as np
from numpy.linalg import eigh, slogdet, inv

def fit_pca_gaussian(train_samples, eps=1e-8):
    mu = train_samples.mean(axis=0)
    X = train_samples - mu
    cov = np.cov(X, rowvar=False)
    evals, evecs = eigh(cov)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    evals = np.maximum(evals, eps)
    return mu, evecs, evals

def sample_pca_gaussian(n_samples, mu, evecs, evals, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_samples, len(evals))) * np.sqrt(evals)
    return z @ evecs.T + mu

def pca_gaussian_nll(samples, mu, evecs, evals):
    X = samples - mu
    Z = X @ evecs
    quad = np.sum((Z ** 2) / evals, axis=1)
    logdet = np.sum(np.log(evals))
    d = samples.shape[1]
    return float(0.5 * np.mean(quad + logdet + d * np.log(2 * np.pi)))
