import numpy as np

def corr_function(samples, max_r=12):
    out = np.zeros(max_r + 1, dtype=np.float64)
    for r in range(max_r + 1):
        out[r] = np.mean(samples * np.roll(samples, -r, axis=1))
    return out

def power_spectrum(samples):
    fft = np.fft.rfft(samples, axis=1)
    return np.mean(np.abs(fft) ** 2, axis=0).real

def corr_err_range(C_model, C_ref, r_start, r_end, eps=1e-12):
    r_end = min(r_end, len(C_ref) - 1)
    if r_start > r_end:
        return np.nan
    denom = abs(C_ref[0]) + eps
    return float(np.mean(np.abs(C_model[r_start:r_end+1] - C_ref[r_start:r_end+1]) / denom))

def spectrum_rel_l2(ps_model, ps_ref, eps=1e-12):
    num = np.linalg.norm(ps_model - ps_ref)
    den = np.linalg.norm(ps_ref) + eps
    return float(num / den)

def estimate_fourier_variances(samples):
    fft = np.fft.rfft(samples, axis=1)
    return np.var(fft.real, axis=0), np.var(fft.imag, axis=0)

def sample_fourier_baseline(n_samples, N, var_real, var_imag, seed=0):
    rng = np.random.default_rng(seed)
    n_modes = N // 2 + 1
    coeffs = np.zeros((n_samples, n_modes), dtype=np.complex128)

    coeffs[:, 0] = rng.normal(0.0, np.sqrt(np.maximum(var_real[0], 1e-12)), size=n_samples)

    if N % 2 == 0:
        coeffs[:, -1] = rng.normal(0.0, np.sqrt(np.maximum(var_real[-1], 1e-12)), size=n_samples)
        kmax = n_modes - 1
    else:
        kmax = n_modes

    for k in range(1, kmax):
        re = rng.normal(0.0, np.sqrt(np.maximum(var_real[k], 1e-12)), size=n_samples)
        im = rng.normal(0.0, np.sqrt(np.maximum(var_imag[k], 1e-12)), size=n_samples)
        coeffs[:, k] = re + 1j * im

    return np.fft.irfft(coeffs, n=N, axis=1).real

def fourier_mode_coupling_normalized(samples):
    fft = np.fft.rfft(samples, axis=1)
    re = fft.real
    im = fft.imag

    re = re - re.mean(axis=0, keepdims=True)
    im = im - im.mean(axis=0, keepdims=True)

    cov_re = np.cov(re, rowvar=False)
    cov_im = np.cov(im, rowvar=False)

    def normalized_ratio(mat):
        diag = np.diag(np.diag(mat))
        off = mat - diag
        return np.linalg.norm(off, ord='fro') / (np.linalg.norm(mat, ord='fro') + 1e-12)

    return normalized_ratio(cov_re), normalized_ratio(cov_im)