import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def loglog_fit(x, y):
    lx = np.log(x)
    ly = np.log(y)
    coef = np.polyfit(lx, ly, 1)
    slope = coef[0]
    intercept = coef[1]
    ly_pred = np.polyval(coef, lx)
    ss_res = np.sum((ly - ly_pred) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    return slope, intercept, r2

def bootstrap_fit(x, y, n_boot=3000, seed=0):
    rng = np.random.default_rng(seed)
    slopes = []
    intercepts = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb = x[idx]
        yb = y[idx]
        try:
            s, b, _ = loglog_fit(xb, yb)
            if np.isfinite(s) and np.isfinite(b):
                slopes.append(s)
                intercepts.append(b)
        except:
            pass
    slopes = np.array(slopes)
    intercepts = np.array(intercepts)
    return {
        "slope_mean": float(np.mean(slopes)),
        "slope_ci_low": float(np.percentile(slopes, 2.5)),
        "slope_ci_high": float(np.percentile(slopes, 97.5)),
        "intercept_mean": float(np.mean(intercepts)),
        "intercept_ci_low": float(np.percentile(intercepts, 2.5)),
        "intercept_ci_high": float(np.percentile(intercepts, 97.5)),
    }

def fit_scaling_from_df(df_one_lam, error_col="fourier_spec_l2"):
    grouped = df_one_lam.groupby("N").agg({
        error_col: ["mean", "std"],
        "re_coupling_norm": ["mean", "std"],
        "im_coupling_norm": ["mean", "std"]
    }).reset_index()

    N_vals = grouped["N"].values.astype(float)
    err_mean = grouped[(error_col, "mean")].values.astype(float)
    err_std = grouped[(error_col, "std")].values.astype(float)
    re_mean = grouped[("re_coupling_norm", "mean")].values.astype(float)
    im_mean = grouped[("im_coupling_norm", "mean")].values.astype(float)
    re_std = grouped[("re_coupling_norm", "std")].values.astype(float)
    im_std = grouped[("im_coupling_norm", "std")].values.astype(float)

    coupling_mean = 0.5 * (re_mean + im_mean)
    coupling_std = 0.5 * np.sqrt(re_std**2 + im_std**2)

    alpha_ec, logA_ec, r2_ec = loglog_fit(coupling_mean, err_mean)
    boot_ec = bootstrap_fit(coupling_mean, err_mean, n_boot=3000, seed=1)

    beta_cn, logB_cn, r2_cn = loglog_fit(N_vals, coupling_mean)
    boot_cn = bootstrap_fit(N_vals, coupling_mean, n_boot=3000, seed=2)

    gamma_en, logC_en, r2_en = loglog_fit(N_vals, err_mean)
    boot_en = bootstrap_fit(N_vals, err_mean, n_boot=3000, seed=3)

    return {
        "grouped": grouped,
        "N_vals": N_vals,
        "err_mean": err_mean,
        "err_std": err_std,
        "coupling_mean": coupling_mean,
        "coupling_std": coupling_std,
        "alpha_ec": alpha_ec,
        "logA_ec": logA_ec,
        "r2_ec": r2_ec,
        "boot_ec": boot_ec,
        "beta_cn": beta_cn,
        "logB_cn": logB_cn,
        "r2_cn": r2_cn,
        "boot_cn": boot_cn,
        "gamma_en": gamma_en,
        "logC_en": logC_en,
        "r2_en": r2_en,
        "boot_en": boot_en,
    }

def plot_baseline_comparisons(df):
    for lam in sorted(df["lam"].unique()):
        sub = df[df["lam"] == lam]
        g = sub.groupby("N").agg({
            "fourier_spec_l2": ["mean", "std"],
            "pca_spec_l2": ["mean", "std"],
            "full_spec_l2": ["mean", "std"],
            "fourier_nll": ["mean", "std"],
            "pca_nll": ["mean", "std"],
            "full_nll": ["mean", "std"],
            "re_coupling_norm": ["mean", "std"],
            "im_coupling_norm": ["mean", "std"]
        }).reset_index()

        N_vals = g["N"].values
        fourier_spec = g[("fourier_spec_l2", "mean")].values
        fourier_spec_std = g[("fourier_spec_l2", "std")].values
        pca_spec = g[("pca_spec_l2", "mean")].values
        pca_spec_std = g[("pca_spec_l2", "std")].values
        full_spec = g[("full_spec_l2", "mean")].values
        full_spec_std = g[("full_spec_l2", "std")].values

        fourier_nll = g[("fourier_nll", "mean")].values
        fourier_nll_std = g[("fourier_nll", "std")].values
        pca_nll = g[("pca_nll", "mean")].values
        pca_nll_std = g[("pca_nll", "std")].values
        full_nll = g[("full_nll", "mean")].values
        full_nll_std = g[("full_nll", "std")].values

        coupling = 0.5 * (
            g[("re_coupling_norm", "mean")].values +
            g[("im_coupling_norm", "mean")].values
        )

        plt.figure(figsize=(6,6))
        plt.errorbar(N_vals, fourier_spec, yerr=fourier_spec_std, marker='o', capsize=4, label='Fourier')
        plt.errorbar(N_vals, pca_spec, yerr=pca_spec_std, marker='s', capsize=4, label='PCA')
        plt.errorbar(N_vals, full_spec, yerr=full_spec_std, marker='^', capsize=4, label='Full Gaussian')
        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.xlabel("System size N", fontsize=10)
        plt.ylabel("Spectral relative error", fontsize=10)
        plt.title(f"Baseline comparison: spec error (lambda={lam})", fontsize=14)
        plt.legend(fontsize=14,)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6,6))
        plt.errorbar(N_vals, fourier_nll, yerr=fourier_nll_std, marker='o', capsize=4, label='Fourier')
        plt.errorbar(N_vals, pca_nll, yerr=pca_nll_std, marker='s', capsize=4, label='PCA')
        plt.errorbar(N_vals, full_nll, yerr=full_nll_std, marker='^', capsize=4, label='Full Gaussian')
        plt.xscale("log", base=2)
        plt.xlabel("System size N", fontsize=10)
        plt.ylabel("Held-out Gaussian NLL", fontsize=10)
        plt.title(f"Baseline comparison: NLL (lambda={lam})", fontsize=14)
        plt.legend(fontsize=14,)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(5,5))
        plt.plot(coupling, fourier_spec, marker='o', label='Fourier')
        plt.plot(coupling, pca_spec, marker='s', label='PCA')
        plt.plot(coupling, full_spec, marker='^', label='Full Gaussian)')
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Average normalized coupling", fontsize=10)
        plt.ylabel("Spectral relative error", fontsize=10)
        plt.title(f"Error vs coupling by baseline (lambda={lam})", fontsize=14)
        plt.legend(fontsize=14,)
        plt.tight_layout()
        plt.show()

def plot_scaling_law_grid(df, error_col="fourier_spec_l2"):
    unique_lams = sorted(df["lam"].unique())
    fig, axes = plt.subplots(len(unique_lams), 3, figsize=(15, 4 * len(unique_lams)))
    if len(unique_lams) == 1:
        axes = np.array([axes])

    for row_idx, lam in enumerate(unique_lams):
        df_lam = df[df["lam"] == lam]
        fit = fit_scaling_from_df(df_lam, error_col=error_col)

        N_vals = fit["N_vals"]
        err_mean = fit["err_mean"]
        err_std = fit["err_std"]
        coupling_mean = fit["coupling_mean"]
        coupling_std = fit["coupling_std"]

        A = np.exp(fit["logA_ec"])
        B = np.exp(fit["logB_cn"])
        C = np.exp(fit["logC_en"])

        xfit1 = np.logspace(np.log10(coupling_mean.min()*0.8), np.log10(coupling_mean.max()*1.2), 300)
        yfit1 = A * xfit1**fit["alpha_ec"]

        xfit2 = np.logspace(np.log10(N_vals.min()*0.9), np.log10(N_vals.max()*1.1), 300)
        yfit2 = B * xfit2**fit["beta_cn"]

        xfit3 = np.logspace(np.log10(N_vals.min()*0.9), np.log10(N_vals.max()*1.1), 300)
        yfit3 = C * xfit3**fit["gamma_en"]

        ax = axes[row_idx, 0]
        ax.errorbar(coupling_mean, err_mean, xerr=coupling_std, yerr=err_std, fmt='o', capsize=4)
        ax.plot(xfit1, yfit1, label=fr'$α={fit["alpha_ec"]:.2f}, R^2={fit["r2_ec"]:.2f}$', color='red')
        for N_val, x, y in zip(N_vals, coupling_mean, err_mean):
            ax.annotate(f"N={int(N_val)}", (x, y), textcoords="offset points", xytext=(4,4))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Average normalized coupling")
        ax.set_ylabel(error_col)
        ax.set_title(f"lambda={lam}: error vs coupling")
        ax.legend()

        ax = axes[row_idx, 1]
        ax.errorbar(N_vals, coupling_mean, yerr=coupling_std, fmt='o', capsize=4)
        ax.plot(xfit2, yfit2, label=fr'$β={fit["beta_cn"]:.2f}, R^2={fit["r2_cn"]:.2f}$', color='red')
        for N_val, x, y in zip(N_vals, N_vals, coupling_mean):
            ax.annotate(f"N={int(N_val)}", (x, y), textcoords="offset points", xytext=(4,4))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("System size N")
        ax.set_ylabel("Average normalized coupling")
        ax.set_title(f"lambda={lam}: coupling vs N")
        ax.legend()

        ax = axes[row_idx, 2]
        ax.errorbar(N_vals, err_mean, yerr=err_std, fmt='o', capsize=4)
        ax.plot(xfit3, yfit3, label=fr'$γ={fit["gamma_en"]:.2f}, R^2={fit["r2_en"]:.2f}$', color='red')
        for N_val, x, y in zip(N_vals, N_vals, err_mean):
            ax.annotate(f"N={int(N_val)}", (x, y), textcoords="offset points", xytext=(4,4))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("System size N")
        ax.set_ylabel(error_col)
        ax.set_title(f"lambda={lam}: error vs N")
        ax.legend()

    plt.tight_layout()
    plt.show()

def make_scaling_summary(df, error_col="fourier_spec_l2"):
    rows = []
    for lam, df_lam in df.groupby("lam"):
        fit = fit_scaling_from_df(df_lam, error_col=error_col)
        rows.append({
            "lam": lam,
            "alpha_error_vs_coupling": fit["alpha_ec"],
            "r2_error_vs_coupling": fit["r2_ec"],
            "alpha_ci_low": fit["boot_ec"]["slope_ci_low"],
            "alpha_ci_high": fit["boot_ec"]["slope_ci_high"],
            "beta_coupling_vs_N": fit["beta_cn"],
            "r2_coupling_vs_N": fit["r2_cn"],
            "beta_ci_low": fit["boot_cn"]["slope_ci_low"],
            "beta_ci_high": fit["boot_cn"]["slope_ci_high"],
            "gamma_error_vs_N": fit["gamma_en"],
            "r2_error_vs_N": fit["r2_en"],
            "gamma_ci_low": fit["boot_en"]["slope_ci_low"],
            "gamma_ci_high": fit["boot_en"]["slope_ci_high"],
        })
    return pd.DataFrame(rows).sort_values("lam")

def plot_gaussianity_vs_coupling(df):
    for lam in sorted(df["lam"].unique()):
        sub = df[df["lam"] == lam]

        g = sub.groupby("N").agg({
            "fourier_kurtosis": ["mean", "std"],
            "fourier_kl": ["mean", "std"],
            "position_kurtosis": ["mean", "std"],
            "position_kl": ["mean", "std"],
            "re_coupling_norm": ["mean"],
            "im_coupling_norm": ["mean"],
        }).reset_index()

        coupling = 0.5 * (
            g[["re_coupling_norm", "mean"]].values +
            g[["im_coupling_norm", "mean"]].values
        )

        fourier_kurt = g[["fourier_kurtosis", "mean"]].values
        fourier_kl = g[["fourier_kl", "mean"]].values
        position_kurt = g[["position_kurtosis", "mean"]].values
        position_kl = g[["position_kl", "mean"]].values

        fig, axes = plt.subplots(figsize=(10,5), ncols=2)

        axes[0].plot(coupling, fourier_kurt, marker='o', label='Fourier Kurtosis')
        axes[0].plot(coupling, position_kurt, marker='s', label='Position Kurtosis') # NEW
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Average normalized coupling")
        axes[0].set_ylabel("Deviation from Gaussianity (Kurtosis)")
        axes[0].set_title(f"Gaussianity breakdown (lambda={lam}) - Kurtosis")
        axes[0].legend()

        axes[1].plot(coupling, fourier_kl, marker='o', label='Fourier KL to Gaussian')
        axes[1].plot(coupling, position_kl, marker='s', label='Position KL to Gaussian') # NEW
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Average normalized coupling")
        axes[1].set_ylabel("Deviation from Gaussianity (KL Divergence)")
        axes[1].set_title(f"Gaussianity breakdown (lambda={lam}) - KL Divergence")
        axes[1].legend()

        plt.tight_layout()
        plt.show()
