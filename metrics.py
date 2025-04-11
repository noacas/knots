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
        self.success_metrics = defaultdict(list)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

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

    def plot_learning_curves(self):
        metrics_df = self.save_metrics(self.metrics, 'metrics')
        success_metrics_df = self.save_metrics(self.success_metrics, 'success_metrics')

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

        # Plot success curve
        plt.figure(figsize=(12, 8))
        cols = [col for col in success_metrics_df.columns if 'success' == col.lower()]

        for col in cols:
            plt.plot(success_metrics_df['step'], success_metrics_df[col], label=col)

        plt.title('Learning Curves - Successes')
        plt.xlabel('Steps')
        plt.ylabel('Success')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, 'success_curves.png'), dpi=300)

        # plot success after how many moves
        plt.figure(figsize=(12, 8))
        cols = [col for col in success_metrics_df.columns if 'success_after_moves' in col.lower()]
        for col in cols:
            plt.plot(success_metrics_df['step'], success_metrics_df[col], label=col)
        plt.title('Learning Curves - Success After Moves')
        plt.xlabel('Steps')
        plt.ylabel('Success After Moves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, 'success_after_moves.png'), dpi=300)


class MetricsStepHook(StepHook):
    def __init__(self, metrics_tracker):
        self.metrics_tracker = metrics_tracker
        super(MetricsStepHook, self).__init__()

    def __call__(self, env, agent, step):
        if env.success:
            self.metrics_tracker.add_success_metric('success_after_moves', env.steps_taken, step)
            self.metrics_tracker.add_success_metric('success', 1, step)
        elif env.done:
            self.metrics_tracker.add_success_metric('success', 0, step)


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