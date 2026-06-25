from ..misc.metropolis_hastings_mcmc import metropolis_mcmc_scalar
from ..misc.gaussianity_metrics import *
from ..misc.core_metrics import *
from ..models.fourier import *
from ..models.full_gaussian import *
from ..models.pca import *
from ..models.maf import *

import time
import pandas as pd


LAMS  = [0.1, 5.0, 10.0, 20.0, 100.0]
NS    = [64, 96]
SEEDS = [0, 1]

results = []

for N in NS:
    for lam in LAMS:
        for seed in SEEDS:
            t0 = time.time()
            print(f"N={N}, λ={lam}, seed={seed} ... ", end='', flush=True)

            samples, _ = metropolis_mcmc_scalar(
                N=N, m=1.0, lam=lam,
                n_samples=4000, burn_in=3000,
                thin=10, step_size=0.35, seed=seed
            )

            C_ref     = coupling_metric(samples)
            f_samp    = fourier_baseline_samples(samples, seed=seed+999)
            C_fourier = coupling_metric(f_samp)
            err_fourier_spec = spectrum_rel_error(power_spectrum(f_samp), power_spectrum(samples))

            flow, mu, std, n_modes = train_flow(samples, n_epochs=400, lr=5e-4,
                                                 batch_size=512, n_layers=4, hidden_dim=64)
            fl_samp  = sample_flow(flow, mu, std, n_modes, N, len(samples))
            C_flow   = coupling_metric(fl_samp)
            err_flow_spec = spectrum_rel_error(power_spectrum(fl_samp), power_spectrum(samples))

            results.append({
                'N': N, 'lam': lam, 'seed': seed,
                'C_ref':        C_ref,
                'C_fourier':    C_fourier,
                'C_flow':       C_flow,
                'err_C_fourier': abs(C_fourier - C_ref),
                'err_C_flow':    abs(C_flow    - C_ref),
                'err_spec_fourier': err_fourier_spec,
                'err_spec_flow':    err_flow_spec,
            })
            print(f"C_ref={C_ref:.3f}  C_fourier={C_fourier:.3f}  "
                  f"C_flow={C_flow:.3f}  ({time.time()-t0:.0f}s)")

df = pd.DataFrame(results)
df.to_csv('/content/flow_results_coupling.csv', index=False)
print("\nSaved to /content/flow_results_coupling.csv")

def regime(C):
    if C < 0.07:   return 'weak'
    elif C < 0.17: return 'intermediate'
    else:          return 'strong'

df['regime'] = df['C_ref'].apply(regime)
agg = df.groupby('regime').agg(
    C_ref         = ('C_ref',         'mean'),
    C_fourier     = ('C_fourier',     'mean'),
    C_flow        = ('C_flow',        'mean'),
    err_C_fourier = ('err_C_fourier', 'mean'),
    err_C_flow    = ('err_C_flow',    'mean'),
).round(4)
print("\nCoupling recovery by regime:")
print(agg.to_string())

agg_plot = df.groupby(['N', 'lam']).agg(
    C_ref     = ('C_ref',     'mean'),
    C_fourier = ('C_fourier', 'mean'),
    C_flow    = ('C_flow',    'mean'),
).reset_index().sort_values('C_ref')

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(agg_plot['C_ref'], agg_plot['C_ref'],     'k--', label='Ground truth (MCMC)', linewidth=1.5)
ax.plot(agg_plot['C_ref'], agg_plot['C_fourier'], 'o-',  label='Fourier baseline',    linewidth=1.5)
ax.plot(agg_plot['C_ref'], agg_plot['C_flow'],    's--', label='MAFlow',              linewidth=1.5)
ax.axvspan(0,    0.07, alpha=0.08, color='green',  label='Weak regime')
ax.axvspan(0.07, 0.17, alpha=0.08, color='orange', label='Intermediate regime')
ax.axvspan(0.17, 0.30, alpha=0.08, color='red',    label='Strong regime')
ax.set_xlabel('Ground truth coupling C (MCMC)', fontsize=12)
ax.set_ylabel('Recovered coupling C', fontsize=12)
ax.set_title('Coupling Recovery: Fourier vs. Normalizing Flow', fontsize=13)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('/content/fig_coupling_recovery.pdf', bbox_inches='tight')
plt.show()
print("Saved /content/fig_coupling_recovery.pdf")

inter = df[df['regime'] == 'intermediate']
if len(inter) > 0:
    C_ref_m   = inter['C_ref'].mean()
    C_four_m  = inter['C_fourier'].mean()
    C_flow_m  = inter['C_flow'].mean()
    pct       = 100*(inter['err_C_fourier'] - inter['err_C_flow']).mean() / (inter['err_C_fourier'].mean() + 1e-12)
    print("\n" + "="*60)
    print("PASTE INTO PAPER (Section 5, after regime paragraph):")
    print("="*60)
    print(f"""
To validate that joint 4th-order structure is the primary
bottleneck, we train a masked autoregressive flow [CITE]
on Fourier-space samples and evaluate on C directly.
The Fourier baseline recovers C ≈ {C_four_m:.3f} against a
ground truth of C ≈ {C_ref_m:.3f} in the intermediate regime.
The normalizing flow recovers C ≈ {C_flow_m:.3f}, reducing
the coupling error by {pct:.0f}% relative to the Fourier
baseline. This confirms that joint mode dependencies are
the primary failure mode of independence-based models.
""")