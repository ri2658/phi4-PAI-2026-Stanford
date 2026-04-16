import numpy as np
from .full_gaussian import gaussian_nll

def fourier_diag_gaussian_nll(samples, var_real, var_imag, eps=1e-8):
    N = samples.shape[1]
    fft = np.fft.rfft(samples, axis=1)
    re = fft.real
    im = fft.imag

    vr = np.maximum(var_real, eps)
    vi = np.maximum(var_imag, eps)

    n_modes = re.shape[1]

    quad = (re[:, 0] ** 2) / vr[0]
    logdet = np.log(vr[0])

    if N % 2 == 0:
        quad += (re[:, -1] ** 2) / vr[-1]
        logdet += np.log(vr[-1])
        upper = n_modes - 1
    else:
        upper = n_modes

    for k in range(1, upper):
        quad += (re[:, k] ** 2) / vr[k] + (im[:, k] ** 2) / vi[k]
        logdet += np.log(vr[k]) + np.log(vi[k])

    d_eff = 1 + (1 if N % 2 == 0 else 0) + 2 * (upper - 1)
    return float(0.5 * np.mean(quad + logdet + d_eff * np.log(2 * np.pi)))

def _get_fourier_components(samples):
    N_original = samples.shape[1]
    fft = np.fft.rfft(samples, axis=1)

    re_parts = fft.real
    im_parts = fft.imag

    # The data vector should be: [real[0], real[1], imag[1], ..., real[(N-1)/2], imag[(N-1)/2]] (N odd)
    # or [real[0], real[1], imag[1], ..., real[N/2-1], imag[N/2-1], real[N/2]] (N even)
    fourier_data = [re_parts[:, 0]]
    for k in range(1, re_parts.shape[1] - (1 if N_original % 2 == 0 else 0)):
        fourier_data.append(re_parts[:, k])
        fourier_data.append(im_parts[:, k])

    if N_original % 2 == 0: # Add the purely real N/2 mode
        fourier_data.append(re_parts[:, -1])

    fourier_components = np.column_stack(fourier_data)

    return fourier_components, N_original

def fit_fourier_full_gaussian(train_samples, eps=1e-6):
    fourier_components, N_original = _get_fourier_components(train_samples)
    mu_fourier = fourier_components.mean(axis=0)
    X_fourier = fourier_components - mu_fourier
    cov_fourier = np.cov(X_fourier, rowvar=False)
    cov_fourier = cov_fourier + eps * np.eye(cov_fourier.shape[0])
    return mu_fourier, cov_fourier, N_original

def sample_fourier_full_gaussian(n_samples, fourier_mu, fourier_cov, N_original, seed=0):
    rng = np.random.default_rng(seed)
    sampled_fourier_components = rng.multivariate_normal(mean=fourier_mu, cov=fourier_cov, size=n_samples)

    # Reconstruct complex Fourier coefficients from the sampled real vector
    n_modes = N_original // 2 + 1
    coeffs = np.zeros((n_samples, n_modes), dtype=np.complex128)

    current_idx = 0
    # k=0 mode (real part)
    coeffs[:, 0] = sampled_fourier_components[:, current_idx]
    current_idx += 1

    # Complex modes k=1 up to n_modes-1 (if N is odd) or n_modes-2 (if N is even, last mode purely real)s
    for k in range(1, n_modes - (1 if N_original % 2 == 0 else 0)):
        coeffs[:, k] = sampled_fourier_components[:, current_idx] + 1j * sampled_fourier_components[:, current_idx + 1]
        current_idx += 2

    # If N is even, the k=N/2 mode is purely real (last component in rfft output)
    if N_original % 2 == 0:
        coeffs[:, n_modes - 1] = sampled_fourier_components[:, current_idx]

    return np.fft.irfft(coeffs, n=N_original, axis=1).real

def fourier_full_gaussian_nll(samples, fourier_mu, fourier_cov):
    fourier_components, _ = _get_fourier_components(samples)
    return gaussian_nll(fourier_components, fourier_mu, fourier_cov)