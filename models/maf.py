import subprocess
subprocess.run(["pip", "install", "nflows", "-q"])

import numpy as np
import math
import time
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
warnings.filterwarnings("ignore")

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

def fourier_baseline_samples(samples, seed=999):
    fft      = np.fft.rfft(samples, axis=1)
    var_real = np.var(fft.real, axis=0)
    var_imag = np.var(fft.imag, axis=0)
    N        = samples.shape[1]
    n_modes  = N // 2 + 1
    n        = len(samples)
    rng      = np.random.default_rng(seed)
    sf       = np.zeros((n, n_modes), dtype=np.complex128)
    sf[:, 0] = rng.normal(0, np.sqrt(np.maximum(var_real[0], 1e-12)), n)
    kmax     = n_modes - 1 if N % 2 == 0 else n_modes
    if N % 2 == 0:
        sf[:, -1] = rng.normal(0, np.sqrt(np.maximum(var_real[-1], 1e-12)), n)
    for k in range(1, kmax):
        sf[:, k] = (rng.normal(0, np.sqrt(np.maximum(var_real[k], 1e-12)), n) +
                    1j * rng.normal(0, np.sqrt(np.maximum(var_imag[k], 1e-12)), n))
    return np.fft.irfft(sf, n=N, axis=1)

def build_flow(dim, n_layers=4, hidden_dim=64):
    transforms = []
    for _ in range(n_layers):
        transforms.append(ReversePermutation(features=dim))
        transforms.append(MaskedAffineAutoregressiveTransform(
            features=dim, hidden_features=hidden_dim,
            num_blocks=2, use_residual_blocks=True, use_batch_norm=False,
        ))
    return Flow(CompositeTransform(transforms), StandardNormal([dim])).to(DEVICE)

def prepare_data(samples):
    fft  = np.fft.rfft(samples, axis=1)
    X    = np.concatenate([fft.real, fft.imag], axis=1).astype(np.float32)
    mu   = X.mean(axis=0, keepdims=True)
    std  = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / std, mu, std, fft.shape[1]

def train_flow(samples, n_epochs=400, lr=5e-4, batch_size=512, n_layers=4, hidden_dim=64):
    X_norm, mu, std, n_modes = prepare_data(samples)
    flow  = build_flow(X_norm.shape[1], n_layers=n_layers, hidden_dim=hidden_dim)
    opt   = optim.Adam(flow.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    X_t   = torch.tensor(X_norm, device=DEVICE)
    flow.train()
    for epoch in range(n_epochs):
        idx  = torch.randperm(len(X_t), device=DEVICE)[:batch_size]
        loss = -flow.log_prob(X_t[idx]).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        opt.step()
        sched.step()
    return flow, mu, std, n_modes

def sample_flow(flow, mu, std, n_modes, N, n_samples):
    flow.eval()
    with torch.no_grad():
        z = flow.sample(n_samples).cpu().numpy()
    z   = z * std + mu
    fft = z[:, :n_modes] + 1j * z[:, n_modes:]
    return np.fft.irfft(fft, n=N, axis=1)