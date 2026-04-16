import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")

def metropolis_mcmc_scalar(N=32, m=1.0, lam=5.0, n_samples=4000, burn_in=3000, thin=10, step_size=0.35, seed=0):
    rng = np.random.default_rng(seed)
    phi = rng.normal(size=(N,)).astype(np.float64)

    def local_action(phi_vec, i):
        im1 = (i - 1) % N
        ip1 = (i + 1) % N
        kinetic = (phi_vec[i] - phi_vec[im1])**2 + (phi_vec[ip1] - phi_vec[i])**2
        mass = (m * m) * phi_vec[i]**2
        quartic = lam * phi_vec[i]**4
        return 0.5 * kinetic + 0.5 * mass + quartic

    total_steps = burn_in + n_samples * thin * N
    out = []
    accepted = 0
    proposed = 0

    for t in range(total_steps):
        i = rng.integers(0, N)
        old = phi[i]
        old_local = local_action(phi, i)

        prop = old + step_size * rng.normal()
        phi[i] = prop
        new_local = local_action(phi, i)

        dS = new_local - old_local
        proposed += 1
        if dS <= 0.0 or rng.random() < math.exp(-dS):
            accepted += 1
        else:
            phi[i] = old

        if t >= burn_in and ((t - burn_in) % (thin * N) == 0):
            out.append(phi.copy())
            if len(out) >= n_samples:
                break

    arr = np.asarray(out, dtype=np.float64)
    arr -= arr.mean()
    return arr, accepted / max(1, proposed)