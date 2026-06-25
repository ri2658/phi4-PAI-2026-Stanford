from misc.metropolis_hastings_mcmc import metropolis_mcmc_scalar
from misc.gaussianity_metrics import *
from misc.core_metrics import *
from models.fourier import *
from models.full_gaussian import *
from models.pca import *
from models.maf import *

import time
import pandas as pd
import numpy as np

def train_test_split_samples(samples, train_frac=0.75):
    n = samples.shape[0]
    n_train = int(train_frac * n)
    return samples[:n_train], samples[n_train:]

def evaluate_model_samples(ref_test, model_samples):
    max_r = min(12, ref_test.shape[1] // 4)
    C_ref = corr_function(ref_test, max_r=max_r)
    C_mod = corr_function(model_samples, max_r=max_r)
    ps_ref = power_spectrum(ref_test)
    ps_mod = power_spectrum(model_samples)

    return {
        "corr_local": corr_err_range(C_mod, C_ref, 0, min(3, max_r)),
        "corr_global": corr_err_range(C_mod, C_ref, 6, max_r),
        "spec_l2": spectrum_rel_l2(ps_mod, ps_ref),
    }

def get_sim_params(N):
    if N == 32:
        return 5000, 3000, 10, 0.35
    if N == 48:
        return 4500, 3000, 10, 0.33
    if N == 64:
        return 4000, 3000, 10, 0.30
    if N == 96:
        return 3000, 3000, 12, 0.27
    return 2500, 3000, 12, 0.25


def regime(C):
    if C < 0.07:
        return 'weak'
    elif C < 0.17:
        return 'intermediate'
    else:
        return 'strong'


def run_one_flow_experiment(N, lam, seed, n_epochs=400, lr=5e-4, batch_size=512, n_layers=4, hidden_dim=64):
    n_samples, burn_in, thin, step_size = get_sim_params(N)

    ref, acc = metropolis_mcmc_scalar(
        N=N, m=1.0, lam=lam,
        n_samples=n_samples, burn_in=burn_in, thin=thin,
        step_size=step_size, seed=seed
    )

    C_ref = coupling_metric(ref)
    f_samp = fourier_baseline_samples(ref, seed=seed + 999)
    C_fourier = coupling_metric(f_samp)
    err_fourier_spec = spectrum_rel_error(power_spectrum(f_samp), power_spectrum(ref))

    flow, mu, std, n_modes = train_flow(
        ref, n_epochs=n_epochs, lr=lr,
        batch_size=batch_size, n_layers=n_layers,
        hidden_dim=hidden_dim
    )
    fl_samp = sample_flow(flow, mu, std, n_modes, N, len(ref))
    C_flow = coupling_metric(fl_samp)
    err_flow_spec = spectrum_rel_error(power_spectrum(fl_samp), power_spectrum(ref))

    return {
        'N': N,
        'lam': lam,
        'seed': seed,
        'acceptance': acc,
        'C_ref': C_ref,
        'C_fourier': C_fourier,
        'C_flow': C_flow,
        'err_C_fourier': abs(C_fourier - C_ref),
        'err_C_flow': abs(C_flow - C_ref),
        'err_spec_fourier': err_fourier_spec,
        'err_spec_flow': err_flow_spec,
    }


def run_flow_experiment(lams=(0.1, 5.0, 10.0, 20.0, 100.0), Ns=(64, 96), seeds=(0, 1), csv_path="/content/flow_results_coupling.csv", n_epochs=400, lr=5e-4, batch_size=512, n_layers=4, hidden_dim=64):
    rows = []
    t0 = time.time()

    for lam in lams:
        for N in Ns:
            for seed in seeds:
                print(f"N={N}, λ={lam}, seed={seed} ... ", end='', flush=True)
                row = run_one_flow_experiment(
                    N=N, lam=lam, seed=seed,
                    n_epochs=n_epochs, lr=lr,
                    batch_size=batch_size,
                    n_layers=n_layers, hidden_dim=hidden_dim
                )
                rows.append(row)
                print(f"C_ref={row['C_ref']:.3f}  C_fourier={row['C_fourier']:.3f}  C_flow={row['C_flow']:.3f}  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")
    return df


def print_flow_summary(df):
    df = df.copy()
    df['regime'] = df['C_ref'].apply(regime)
    agg = df.groupby('regime').agg(
        C_ref=('C_ref', 'mean'),
        C_fourier=('C_fourier', 'mean'),
        C_flow=('C_flow', 'mean'),
        err_C_fourier=('err_C_fourier', 'mean'),
        err_C_flow=('err_C_flow', 'mean')
    ).round(4)
    print("\nCoupling recovery by regime:")
    print(agg.to_string())
    return agg


def run_one_experiment(N, lam, seed, num_fourier_blocks=4):
    n_samples, burn_in, thin, step_size = get_sim_params(N)

    ref, acc = metropolis_mcmc_scalar(
        N=N, m=1.0, lam=lam,
        n_samples=n_samples, burn_in=burn_in, thin=thin,
        step_size=step_size, seed=seed
    )

    train_samples, test_samples = train_test_split_samples(ref, train_frac=0.75)

    var_real, var_imag = estimate_fourier_variances(train_samples)
    fourier_samples = sample_fourier_baseline(len(test_samples), N, var_real, var_imag, seed=seed + 100)
    fourier_eval = evaluate_model_samples(test_samples, fourier_samples)
    fourier_nll = fourier_diag_gaussian_nll(test_samples, var_real, var_imag)

    pca_mu, pca_vecs, pca_vals = fit_pca_gaussian(train_samples)
    pca_samples = sample_pca_gaussian(len(test_samples), pca_mu, pca_vecs, pca_vals, seed=seed + 200)
    pca_eval = evaluate_model_samples(test_samples, pca_samples)
    pca_nll = pca_gaussian_nll(test_samples, pca_mu, pca_vecs, pca_vals)

    full_mu, full_cov = fit_full_gaussian(train_samples)
    full_samples = sample_full_gaussian(len(test_samples), full_mu, full_cov, seed=seed + 300)
    full_eval = evaluate_model_samples(test_samples, full_samples)
    full_nll = gaussian_nll(test_samples, full_mu, full_cov)

    
    re_ratio, im_ratio = fourier_mode_coupling_normalized(train_samples)

    fft = np.fft.rfft(train_samples, axis=1)
    fft_samples = np.concatenate([fft.real, fft.imag], axis=1)

    fourier_kurtosis = compute_kurtosis(fft_samples)
    fourier_kl = kl_to_gaussian_1d(fft_samples)

    position_kurtosis = compute_kurtosis(train_samples)
    position_kl = kl_to_gaussian_1d(train_samples)

    return {
        "N": N,
        "lam": lam,
        "seed": seed,
        "acceptance": acc,

        "fourier_corr_local": fourier_eval["corr_local"],
        "fourier_corr_global": fourier_eval["corr_global"],
        "fourier_spec_l2": fourier_eval["spec_l2"],
        "fourier_nll": fourier_nll,

        "pca_corr_local": pca_eval["corr_local"],
        "pca_corr_global": pca_eval["corr_global"],
        "pca_spec_l2": pca_eval["spec_l2"],
        "pca_nll": pca_nll,

        "full_corr_local": full_eval["corr_local"],
        "full_corr_global": full_eval["corr_global"],
        "full_spec_l2": full_eval["spec_l2"],
        "full_nll": full_nll,

        "re_coupling_norm": re_ratio,
        "im_coupling_norm": im_ratio,

        "fourier_kurtosis": fourier_kurtosis,
        "fourier_kl": fourier_kl,
        "position_kurtosis": position_kurtosis,
        "position_kl": position_kl,
    }

def run_full_experiment(lams=(5.0, 10.0, 20.0), Ns=(32, 48, 64, 96, 128), seeds=(0, 1, 2), num_fourier_blocks=4, csv_path="/content/full_experiment.csv"):
    rows = []
    t0 = time.time()

    for lam in lams:
        print(f"\n===== lambda = {lam} ===")
        for N in Ns:
            for seed in seeds:
                print(f"Running N={N}, lam={lam}, seed={seed}")
                rows.append(run_one_experiment(N, lam, seed, num_fourier_blocks)) # Pass num_fourier_blocks

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nFinished in {time.time()-t0:.1f}s")
    print("Saved CSV to:", csv_path)
    return df

def print_summary_tables(df):
    summary = df.groupby(["lam", "N"]).mean(numeric_only=True)
    print(summary)
    return summary