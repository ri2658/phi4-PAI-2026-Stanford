import pandas as pd
import argparse
from .training.training_loop import *
from .plotting.plotting import *

parser = argparse.ArgumentParser(description="Run full experiment with customizable parameters.")
parser.add_argument('--lams', nargs='+', type=float, default=[0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0], help='List of lambda values')
parser.add_argument('--Ns', nargs='+', type=int, default=[32, 48, 64, 96, 128], help='List of N values')
parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2], help='List of seed values')
parser.add_argument('--csv_path', type=str, default="/content/full_experiment.csv", help='Path to save/load the CSV file')
parser.add_argument('--num_fourier_blocks', type=int, default=4, help='Number of Fourier blocks')

args = parser.parse_args()

try:
    df = pd.read_csv(args.csv_path)
except FileNotFoundError:
    print("Previous data not found, so starting default experiment.")
    df = run_full_experiment(
        lams=tuple(args.lams),
        Ns=tuple(args.Ns),
        seeds=tuple(args.seeds),
        csv_path=args.csv_path,
        num_fourier_blocks=args.num_fourier_blocks
    )

summary = print_summary_tables(df)
print("\nScaling summary for Fourier spectral error:")
print(make_scaling_summary(df, error_col="fourier_spec_l2"))

plot_baseline_comparisons(df)
plot_scaling_law_grid(df, error_col="fourier_spec_l2")

plot_gaussianity_vs_coupling(df)