import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import os

from pfrl.experiments import EvaluationHook
from pfrl.experiments.hooks import StepHook


# Create a metrics tracker class
class MetricsTracker:
    def __init__(self, save_dir):
        self.metrics = defaultdict(list)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def add_metric(self, name, value, step):
        self.metrics[name].append((step, value))

    def save_metrics(self):
        metrics_df = pd.DataFrame()
        for name, values in self.metrics.items():
            steps, metric_values = zip(*values)
            temp_df = pd.DataFrame({'step': steps, name: metric_values})
            if metrics_df.empty:
                metrics_df = temp_df
            else:
                metrics_df = pd.merge(metrics_df, temp_df, on='step', how='outer')

        metrics_df.to_csv(os.path.join(self.save_dir, 'metrics.csv'), index=False)
        return metrics_df

    def plot_learning_curves(self):
        metrics_df = self.save_metrics()

        # Plot reward curves
        plt.figure(figsize=(12, 8))
        reward_cols = [col for col in metrics_df.columns if 'reward' in col.lower()]

        for col in reward_cols:
            plt.plot(metrics_df['step'], metrics_df[col], label=col)

        plt.title('Learning Curves - Rewards')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, 'reward_curves.png'), dpi=300)

        # Plot loss curves
        plt.figure(figsize=(12, 8))
        loss_cols = [col for col in metrics_df.columns if 'loss' in col.lower()]

        for col in loss_cols:
            plt.plot(metrics_df['step'], metrics_df[col], label=col)

        plt.title('Learning Curves - Losses')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, 'loss_curves.png'), dpi=300)


class MetricsStepHook(StepHook):
    def __init__(self, metrics_tracker):
        self.metrics_tracker = metrics_tracker
        super(MetricsStepHook, self).__init__()

    def __call__(self, env, agent, step):
        if env.success:
            self.metrics_tracker.add_metric('success', step, step)


class MetricsEvaluationHook(EvaluationHook):
    def __init__(self, metrics_tracker, outdir):
        self.metrics_tracker = metrics_tracker
        self.outdir = outdir
        self.support_train_agent = True
        self.support_train_agent_batch = False
        self.support_train_agent_async = False
        super().__init__()

    def __call__(self, env, agent, evaluator, step, eval_stats, agent_stats, env_stats):
        # Add metrics from evaluation
        self.metrics_tracker.add_metric('eval_mean_reward', eval_stats['mean'], step)
        self.metrics_tracker.add_metric('eval_median_reward', eval_stats['median'], step)
        success_rate = sum(1 for r in eval_stats['rewards'] if r > 0) / len(eval_stats['rewards'])
        self.metrics_tracker.add_metric('eval_success_rate', success_rate, step)

        # Plot current learning curves
        self.metrics_tracker.plot_learning_curves()

        # Save agent checkpoint
        agent.save(os.path.join(self.outdir, f'agent_step_{step}'))

        return eval_stats