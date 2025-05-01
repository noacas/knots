import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import os

from stable_baselines3.common.callbacks import BaseCallback

from braid_env import BraidEnvironment


# Create a metrics tracker class
def create_success_rate_df(df):
    # Group by step and calculate success metrics
    df["rounded_step"] = df["step"].round(-3)
    success_by_step = df.groupby('rounded_step')['success'].agg(['count', 'sum']).reset_index()

    # Calculate success rate
    success_by_step['success_rate'] = success_by_step['sum'] / success_by_step['count'] * 100

    # Rename columns for clarity
    success_by_step.rename(columns={
        'rounded_step': 'step',
        'count': 'total_attempts',
        'sum': 'successful_attempts'
    }, inplace=True)

    # Add rolling average to smooth out the data (optional)
    if len(success_by_step) >= 10:
        success_by_step['avg_success_rate'] = success_by_step['success_rate'].rolling(window=1000, center=False).mean()
    else:
        # Add an empty column if not enough data points
        success_by_step['avg_success_rate'] = None

    # Return only the specified columns
    return success_by_step[["step", "success_rate", "avg_success_rate"]]


class MetricsTracker:
    def __init__(self, save_dir, save_interval=1000):
        self.metrics = defaultdict(list)
        self.success_metrics = defaultdict(list)
        self.save_dir = save_dir
        self.last_save_step = 0
        self.save_interval = save_interval
        os.makedirs(save_dir, exist_ok=True)

    def should_save(self, step):
        if step - self.last_save_step >= self.save_interval:
            self.last_save_step = step
            return True
        return False

    def add_metric(self, name, value, step):
        self.metrics[name].append((step, value))

    def add_success_metric(self, name, value, step):
        self.success_metrics[name].append((step, value))

    def save_metrics(self, metrics, file_name):
        metrics_df = pd.DataFrame()
        for name, values in metrics.items():
            steps, metric_values = zip(*values)
            temp_df = pd.DataFrame({'step': steps, name: metric_values})
            if metrics_df.empty:
                metrics_df = temp_df
            else:
                metrics_df = pd.merge(metrics_df, temp_df, on='step', how='outer')

        metrics_df.to_csv(os.path.join(self.save_dir, f'{file_name}.csv'), index=False)
        return metrics_df

    def plot_learning_curves(self, step=0):
        if len(self.success_metrics) == 0:
            return

        self.last_save_step = step
        success_metrics_df = self.save_metrics(self.success_metrics, 'success_metrics')

        # Plot success curve
        success_rate_df = create_success_rate_df(success_metrics_df)

        plt.figure(figsize=(12, 8))
        for col in success_rate_df.columns:
            if col != "step":
                plt.plot(success_rate_df['step'], success_rate_df[col], label=col)

        plt.title('Learning Curves - Successes')
        plt.xlabel('Steps')
        plt.ylabel('Success')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, 'success_curves.png'), dpi=300)

        # plot success after how many moves
        plt.figure(figsize=(12, 8))
        success_after_moves = success_metrics_df[success_metrics_df['success_after_moves'].notnull()]
        plt.plot(success_after_moves['step'], success_after_moves['success_after_moves'], label='success_after_moves')
        plt.title('Learning Curves - Success After Moves')
        plt.xlabel('Steps')
        plt.ylabel('Success After Moves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, 'success_after_moves.png'), dpi=300)


class MetricsStepHook(BaseCallback):

    def __init__(self, metrics_tracker: MetricsTracker, env: BraidEnvironment):
        super(MetricsStepHook, self).__init__()
        self.metrics_tracker = metrics_tracker
        self.env = env

    def _on_step(self) -> bool:
        if self.env.success:
            self.metrics_tracker.add_success_metric('success_after_moves', self.env.steps_taken, self.n_calls)
            self.metrics_tracker.add_success_metric('success', 1, self.n_calls)
        elif self.env.done:
            self.metrics_tracker.add_success_metric('success', 0, self.n_calls)
        return True

    def _on_training_end(self) -> None:
        self.metrics_tracker.plot_learning_curves()