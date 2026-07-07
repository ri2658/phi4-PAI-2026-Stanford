# When Independent Gaussian Models Break Down: Characterizing Regime-Dependent Modeling Failures in $\phi^4$ Theory: Official Repository

This project implements generative machine learning models for the φ⁴ scalar field theory in lattice quantum field theory. It uses Metropolis-Hastings Markov Chain Monte Carlo (MCMC) to generate ground truth samples and trains various generative models (Fourier Neural Networks, Principal Component Analysis, and Full Gaussian approximations) to learn the distribution.

## Overview

The φ⁴ theory is a fundamental model in quantum field theory that describes scalar fields with quartic interactions. This project explores how modern machine learning techniques can approximate the complex probability distributions arising from lattice simulations of this theory.

The code performs systematic experiments across different coupling strengths (λ) and lattice sizes (N), evaluating model performance through correlation functions and power spectra.

## Features

- **MCMC Sampling**: Metropolis-Hastings algorithm for generating φ⁴ field configurations
- **Generative Models**:
  - Fourier Neural Networks for spectral domain modeling
  - PCA-based dimensionality reduction
  - Full Gaussian approximation as baseline
- **Evaluation Metrics**:
  - Correlation function errors (local and global ranges)
  - Power spectrum L² relative errors
  - Coupling recovery and MAF flow evaluation against Fourier baselines
- **Comprehensive Analysis**: Scaling laws, gaussianity metrics, and visualization tools
- **Experiment Management**: CSV-based result tracking and resumable experiments

## Coupling diagnostic → `couplnorm`

The normalized fourth-order coupling metric **C** used here — the covariance of
spectral energies `E_k = |φ̃_k|²` reduced to `‖Σ − diag(Σ)‖_F / ‖Σ‖_F` — is
packaged as a standalone, tested PyTorch library:
**[couplnorm](https://github.com/anishbhat28/couplnorm)**
(`CouplingMetric`, `CouplingLoss`, `coupling_from_samples`). It computes the same
number as `misc/core_metrics.coupling_metric` (verified to ~1e-8 on identical
samples) and additionally exposes C as a differentiable training loss.

**FFT convention (important).** C is computed on the `N//2 + 1` unique modes of
the real FFT (`numpy.fft.rfft`). For a real field the full FFT is
conjugate-symmetric (`|φ̃_k|² = |φ̃_{N−k}|²`), which would force `N − 2`
off-diagonal covariance entries to equal diagonal ones and inflate C to a
physics-independent floor of `√((N−2)/(2N−2)) ≈ 0.7`. Restricting to unique modes
(as this code does) means the reported values (C ∈ [0.06, 0.2]) reflect genuine
mode coupling rather than a symmetry artifact.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/phi4-PAI-2026-Stanford.git
   cd phi4-PAI-2026-Stanford
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scipy torch nflows
   ```
   (Add other dependencies as needed based on your environment)

## Usage

Run the full experiment suite:

```bash
python main.py
```

Run the flow coupling experiment via the main runner:

```bash
python main.py --run_flow_experiment
```

Alternatively, run the dedicated flow experiment script directly:

```bash
python training/maf_exp.py
```

Customize parameters for the main experiment:

```bash
python main.py --lams 0.1 1.0 5.0 --Ns 32 64 128 --seeds 0 1 2 --num_fourier_blocks 4
```

### Command Line Arguments

- `--lams`: List of coupling constants λ (default: [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0])
- `--Ns`: List of lattice sizes N (default: [32, 48, 64, 96, 128])
- `--seeds`: Random seeds for reproducibility (default: [0, 1, 2])
- `--csv_path`: Path for saving/loading experiment results (default: "/content/full_experiment.csv")
- `--num_fourier_blocks`: Number of Fourier neural network blocks (default: 4)
- `--run_flow_experiment`: Run the masked autoregressive flow coupling experiment (default: false)
- `--flow_csv_path`: Path for saving/loading the flow experiment CSV file (default: "/content/flow_results_coupling.csv")
- `--flow_n_epochs`: Number of flow training epochs (default: 400)
- `--flow_lr`: Learning rate for flow training (default: 5e-4)
- `--flow_batch_size`: Batch size for flow training (default: 512)
- `--flow_n_layers`: Number of flow layers (default: 4)
- `--flow_hidden_dim`: Hidden dimension in flow layers (default: 64)

## Project Structure

```
├── main.py                 # Main experiment runner
├── misc/
│   ├── core_metrics.py     # Correlation and spectrum calculations
│   ├── gaussianity_metrics.py  # Gaussianity analysis
│   └── metropolis_hastings_mcmc.py  # MCMC sampling implementation
├── models/
│   ├── fourier.py          # Fourier neural network model
│   ├── full_gaussian.py    # Gaussian baseline model
│   └── pca.py              # PCA-based model
├── plotting/
│   └── plotting.py         # Visualization functions
├── training/
│   ├── training_loop.py    # Training and evaluation logic
│   └── maf_exp.py          # Direct flow experiment script
├── LICENSE                 # MIT License
└── README.md              # This file
```

## Results and Analysis

The project generates:
- Summary tables of model performance
- Scaling law plots showing how errors depend on lattice size
- Gaussianity analysis vs coupling strength
- Baseline model comparisons

Results are saved to CSV for further analysis.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Ryo I.
- Anish B.
- Zihan Z.

## Acknowledgments

This work was developed for 2026 Conference on Physics and AI (PAI26) at Stanford University, 2026.
