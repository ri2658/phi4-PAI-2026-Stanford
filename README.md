# When Independent Gaussian Models Break Down: Characterizing Regime-Dependent Modeling Failures in φ⁴ Theory 

This project implements generative machine learning models for the φ⁴ scalar field theory in lattice quantum field theory. It uses Metropolis-Hastings Markov Chain Monte Carlo (MCMC) to generate ground truth samples and uses various methods (Fourier Representations, Principal Component Analysis, and Full Gaussian approximations) to represent  the distribution.

## Overview

The φ⁴ theory is a fundamental model in quantum field theory that describes scalar fields with quartic interactions. This project explores how modern machine learning techniques can approximate the complex probability distributions arising from lattice simulations of this theory.

The code performs systematic experiments across different coupling strengths (λ) and lattice sizes (N), evaluating model performance through correlation functions and power spectra.

## Features

- **MCMC Sampling**: Metropolis-Hastings algorithm for generating φ⁴ field configurations
- **Generative Models**:
  - Fourier Representation for spectral domain modeling
  - PCA-based dimensionality reduction
  - Full Gaussian approximation as baseline
- **Evaluation Metrics**:
  - Correlation function errors (local and global ranges)
  - Power spectrum L² relative errors
- **Comprehensive Analysis**: Scaling laws, gaussianity metrics, and visualization tools
- **Experiment Management**: CSV-based result tracking and resumable experiments

## Usage

Run the full experiment suite:

```bash
python main.py
```

Customize parameters:

```bash
python main.py --lams 0.1 1.0 5.0 --Ns 32 64 128 --seeds 0 1 2 --num_fourier_blocks 4
```

### Command Line Arguments

- `--lams`: List of coupling constants λ (default: [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0])
- `--Ns`: List of lattice sizes N (default: [32, 48, 64, 96, 128])
- `--seeds`: Random seeds for reproducibility (default: [0, 1, 2])
- `--csv_path`: Path for saving/loading experiment results (default: "/content/full_experiment.csv")
- `--num_fourier_blocks`: Number of Fourier neural network blocks (default: 4)

## Project Structure

```
├── main.py                 # Main experiment runner
├── misc/
│   ├── core_metrics.py     # Correlation and spectrum calculations
│   ├── gaussianity_metrics.py  # Gaussianity analysis
│   └── metropolis_hastings_mcmc.py  # MCMC sampling implementation
├── models/
│   ├── fourier.py          # Fourier model
│   ├── full_gaussian.py    # Gaussian baseline model
│   └── pca.py              # PCA-based model
├── plotting/
│   └── plotting.py         # Visualization functions
├── training/
│   └── training_loop.py    # Training and evaluation logic
├── LICENSE                 # MIT License
└── README.md               # This file
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


## Acknowledgments

This work was developed for 2026 Conference on Physics and AI (PAI26) at Stanford University, 2026.
