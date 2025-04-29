import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def create_training_progression_graphs(file_path, output_dir="./graphs"):
    """
    Generate visualizations showing the progression of training metrics over time.

    Parameters:
    -----------
    file_path : str
        Path to the tab-separated text file with multiple rows of training data
    output_dir : str
        Directory to save the generated graphs
    """
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Read the tab-separated file - assuming multiple rows
    # If your file has only one header line and multiple data rows:
    df = pd.read_csv(file_path, sep='\t')

    # If your actual file has multiple entries with repeating headers,
    # you might need more complex parsing logic

    # Sort by steps to ensure chronological order
    if 'steps' in df.columns:
        df = df.sort_values('steps')

    # Set figure style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})

    # 1. Policy Convergence - Track entropy over time
    plt.figure(figsize=(12, 6))
    plt.plot(df['steps'], df['average_entropy'], marker='o', linestyle='-',
             color='blue', linewidth=2, markersize=8)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7,
                label='Convergence threshold (0.5)')
    plt.title('Policy Convergence: Entropy Reduction Over Training', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Average Entropy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add annotation for key insight
    final_entropy = df['average_entropy'].iloc[-1]
    initial_entropy = df['average_entropy'].iloc[0]
    plt.annotate(f'Initial entropy: {initial_entropy:.2f}\nFinal entropy: {final_entropy:.2f}',
                 xy=(df['steps'].iloc[-1], final_entropy),
                 xytext=(df['steps'].iloc[-1] * 0.8, initial_entropy * 0.8),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/policy_convergence.png", dpi=300)

    # 2. Value Function Accuracy - Track explained variance over time
    plt.figure(figsize=(12, 6))
    plt.plot(df['steps'], df['explained_variance'], marker='o', linestyle='-',
             color='green', linewidth=2, markersize=8)
    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.7,
                label='High accuracy threshold (0.95)')
    plt.title('Value Function Accuracy: Explained Variance Over Training', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Explained Variance', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add annotation for improvement
    final_var = df['explained_variance'].iloc[-1]
    initial_var = df['explained_variance'].iloc[0]
    improvement = final_var - initial_var
    plt.annotate(f'Improvement: {improvement:.2f}',
                 xy=(df['steps'].iloc[-1], final_var),
                 xytext=(df['steps'].iloc[-1] * 0.7, final_var * 0.9),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/value_function_accuracy.png", dpi=300)

    # 3. KL Divergence Control - Track KL divergence with TRPO threshold
    plt.figure(figsize=(12, 6))
    plt.plot(df['steps'], df['average_kl'], marker='o', linestyle='-',
             color='purple', linewidth=2, markersize=8)
    plt.axhline(y=0.01, color='r', linestyle='--', alpha=0.7,
                label='TRPO threshold (0.01)')
    plt.title('KL Divergence Control During Policy Updates', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Average KL Divergence', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add annotation for stability
    max_kl = df['average_kl'].max()
    below_threshold = df['average_kl'].lt(0.01).mean() * 100

    plt.annotate(f'Below threshold: {below_threshold:.1f}% of updates\nMax KL: {max_kl:.4f}',
                 xy=(df['steps'].iloc[-1], df['average_kl'].iloc[-1]),
                 xytext=(df['steps'].iloc[-1] * 0.7, 0.01 * 1.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/kl_divergence_control.png", dpi=300)

    # 4. Combined metrics visualization (all three metrics normalized)
    plt.figure(figsize=(14, 7))

    # Normalize metrics for comparison
    df_norm = df.copy()
    metrics = ['average_entropy', 'explained_variance', 'average_kl']

    for metric in metrics:
        max_val = df[metric].max()
        min_val = df[metric].min()
        # Handle cases where max = min
        if max_val == min_val:
            df_norm[f"{metric}_norm"] = 0.5  # Set to middle of range
        else:
            df_norm[f"{metric}_norm"] = (df[metric] - min_val) / (max_val - min_val)

    # Plot normalized metrics on the same graph
    plt.plot(df_norm['steps'], df_norm['average_entropy_norm'],
             label='Entropy (normalized)', linestyle='-', marker='o', color='blue')
    plt.plot(df_norm['steps'], df_norm['explained_variance_norm'],
             label='Explained Variance (normalized)', linestyle='-', marker='s', color='green')
    plt.plot(df_norm['steps'], df_norm['average_kl_norm'],
             label='KL Divergence (normalized)', linestyle='-', marker='^', color='purple')

    plt.title('Combined Training Metrics Progression', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Normalized Value', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/combined_metrics.png", dpi=300)

    # 5. Create a scatter plot showing relationship between entropy and explained variance
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['average_entropy'], df['explained_variance'],
                          c=df['steps'], cmap='viridis', s=100, alpha=0.8)

    plt.colorbar(scatter, label='Training Steps')
    plt.title('Relationship: Policy Convergence vs Value Function Accuracy', fontsize=16)
    plt.xlabel('Average Entropy (Policy Convergence)', fontsize=14)
    plt.ylabel('Explained Variance (Value Function Accuracy)', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add arrows to show progression direction
    for i in range(len(df) - 1):
        plt.annotate('',
                     xy=(df['average_entropy'].iloc[i + 1], df['explained_variance'].iloc[i + 1]),
                     xytext=(df['average_entropy'].iloc[i], df['explained_variance'].iloc[i]),
                     arrowprops=dict(facecolor='red', width=1, alpha=0.6, headwidth=8))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/entropy_vs_variance.png", dpi=300)

    print(f"Generated 5 training progression graphs in directory: {output_dir}")


# Example usage
if __name__ == "__main__":
    create_training_progression_graphs("scores.txt")