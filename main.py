import argparse
import pandas as pd
from training.training_loop import *
from plotting.plotting import *


def main():
    parser = argparse.ArgumentParser(description="Run full experiment with customizable parameters.")
    parser.add_argument('--lams', nargs='+', type=float, default=[0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0], help='List of lambda values')
    parser.add_argument('--Ns', nargs='+', type=int, default=[32, 48, 64, 96, 128], help='List of N values')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2], help='List of seed values')
    parser.add_argument('--csv_path', type=str, default="/content/full_experiment.csv", help='Path to save/load the CSV file')
    parser.add_argument('--num_fourier_blocks', type=int, default=4, help='Number of Fourier blocks')
    parser.add_argument('--run_flow_experiment', action='store_true', help='Run the masked autoregressive flow coupling experiment')
    parser.add_argument('--flow_csv_path', type=str, default="/content/flow_results_coupling.csv", help='Path to save/load the flow experiment CSV file')
    parser.add_argument('--flow_n_epochs', type=int, default=400, help='Number of flow training epochs')
    parser.add_argument('--flow_lr', type=float, default=5e-4, help='Learning rate for flow training')
    parser.add_argument('--flow_batch_size', type=int, default=512, help='Batch size for flow training')
    parser.add_argument('--flow_n_layers', type=int, default=4, help='Number of flow layers')
    parser.add_argument('--flow_hidden_dim', type=int, default=64, help='Hidden dimension in flow layers')

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

    if args.run_flow_experiment:
        try:
            df_flow = pd.read_csv(args.flow_csv_path)
        except FileNotFoundError:
            print("Previous flow data not found, so starting flow experiment.")
            df_flow = run_flow_experiment(
                lams=tuple(args.lams),
                Ns=tuple(args.Ns),
                seeds=tuple(args.seeds),
                csv_path=args.flow_csv_path,
                n_epochs=args.flow_n_epochs,
                lr=args.flow_lr,
                batch_size=args.flow_batch_size,
                n_layers=args.flow_n_layers,
                hidden_dim=args.flow_hidden_dim,
            )

        print("\nFlow experiment summary:")
        print_flow_summary(df_flow)


if __name__ == '__main__':
    main()
